import os
from datetime import datetime
from pathlib import Path

import global_utils.global_core as g
import matplotlib
import numpy as np
import torch
from dataset_utils.baseline_dataset import BaselineDataSet
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetPart, DatasetVer, ErrMsg, Metric, Stat
from loss_utils.unified_focal_loss import UnifiedFocalLoss

# Prevent matplotlib.pyplot from using a GUI (like X11) for rendering.
# Without this line, using breakpoints under X11 without VCXSRV can cause the debugger to freeze.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from numpy import ndarray
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_utils.training_core import TrainingCore


class BaselineTraining(TrainingCore):
    def __init__(self):
        super().__init__()

    def _load_hyper(
        self,
        hyper: Dict,
        device_id: int,
        idl_gtvn_baseline_id: str,
        fold: int,
        debug_mode: bool,
    ):
        # shared by baseline/idl.gtvn/idl.gtvt
        super()._load_hyper(hyper=hyper, device_id=device_id)

        # epochs
        if debug_mode:
            # only train 2 epoch in debug mode
            hyper["epochs"] = 2
        else:
            hyper["epochs"] = g.clamp_value(hyper["epochs"], (1, None))

        # record actual epochs because of early stop
        hyper["epochs.actual"] = 0

        # early stop, based on epoch
        hyper["early.stop.epochs"] = g.clamp_value(
            hyper["early.stop.epochs"], (1, hyper["epochs"])
        )

        # lr
        hyper["lr"] = g.clamp_value(hyper["lr"], (g.EPS, 1.0))

        # actual lr
        used_gpu_count = g.used_gpu_count(device_id)
        if used_gpu_count > 1:
            hyper["lr.actual"] = hyper["lr"] * used_gpu_count
        else:
            hyper["lr.actual"] = hyper["lr"]

        # min lr
        hyper["lr.min"] = g.clamp_value(hyper["lr.min"], (g.EPS, hyper["lr"]))

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        hyper["lr.decay.patience"] = g.clamp_value(
            hyper["lr.decay.patience"], (1, hyper["epochs"])
        )

        # number of best valid loss cnn retained
        hyper["keep.best.cnn.num"] = g.clamp_value(
            hyper["keep.best.cnn.num"], (1, hyper["epochs"])
        )

        # augment percent
        hyper["augment.pct"] = g.clamp_value(hyper["augment.pct"], (0.0, 1.0))

        self._load_hyper_no_pt_mr(
            hyper=hyper,
            idl_gtvn_baseline_id=idl_gtvn_baseline_id,
        )

        # use geodesic distance or not
        if hyper["geodesic.distance"] == {}:
            hyper.pop("geodesic.distance")
        else:
            hyper["geodesic.distance"] = bool(hyper["geodesic.distance"])

        # load patients after dataset version is selected
        patients = self._load_patients(
            dataset_ver=hyper["dataset.ver"],
            fold=fold,
            debug_mode=debug_mode,
        )
        for key_name in [DatasetPart.TRAIN, DatasetPart.VALID]:
            hyper["{}.patients".format(key_name)] = patients[key_name]

        self._load_hyper_loss_func(hyper=hyper, device_id=device_id)

        # load datasets before load dataloaders
        self._load_hyper_data_sets(hyper)

        self._load_hyper_data_loaders(hyper)

        # load cnn (before optimizer)
        # (1) normal training
        if (
            hyper["pretrain.id"] == ""
            or hyper["pretrain.id"] is None
            or hyper["pretrain.id"] == {}
        ):
            hyper["pretrain.id"] = None
            self._load_hyper_new_cnn(hyper=hyper, device_id=device_id)
        # (2) transfer learning
        else:
            pretrain_cnn_path = self._find_best_cnn_in_folds(hyper["pretrain.id"])
            pretrain_hyper = g.load_json(
                os.path.join(Path(pretrain_cnn_path).parent.parent, "hyper.json")
            )
            pretrain_no_pt = True if pretrain_hyper["no.pt"] else False
            if hyper["no.pt"] != pretrain_no_pt:
                g.error_exit(
                    "'no.pt' hyper mismatch between existing CNN and transfer learning!"
                )
            pretrain_no_mr = True if pretrain_hyper["no.mr"] else False
            if hyper["no.mr"] != pretrain_no_mr:
                g.error_exit(
                    "'no.mr' hyper mismatch between existing CNN and transfer learning!"
                )
            hyper["cnn"] = self._load_exist_cnn(
                cnn_path=pretrain_cnn_path,
                device_id=device_id,
            )

        self._load_hyper_optim_and_scheduler(hyper=hyper)

    def _load_hyper_no_pt_mr(self, hyper: Dict, idl_gtvn_baseline_id: str):
        # MDA dataset has no PET imgs
        if hyper["dataset.ver"] == DatasetVer.MDA:
            hyper["no.pt"] = True
        # HECKTOR dataset has no MR-T1 and MR-T2 imgs
        elif hyper["dataset.ver"] == DatasetVer.HECKTOR:
            hyper["no.mr"] = True

        # idl gtvn
        if idl_gtvn_baseline_id is not None:
            baseline_dir = self._find_train_dir(idl_gtvn_baseline_id)
            baseline_fold_dirs = g.get_sub_dirs(
                baseline_dir, key_word="fold=", full_path=True
            )
            baseline_hyper = g.load_json(
                os.path.join(baseline_fold_dirs[0], "hyper.json")
            )
            baseline_no_pt = True if baseline_hyper["no.pt"] else False
            if hyper["no.pt"] != baseline_no_pt:
                g.error_exit("'no.pt' hyper mismatch between idl.gtvn and baseline!")
            baseline_no_mr = True if baseline_hyper["no.mr"] else False
            if hyper["no.mr"] != baseline_no_mr:
                g.error_exit("'no.mr' hyper mismatch between idl.gtvn and baseline!")

    def _load_hyper_new_cnn(
        self,
        hyper: Dict,
        device_id: int,
        in_chan: int = 4,
        out_chan: int = 3,
    ):
        if hyper["no.pt"]:
            in_chan -= 1
        if hyper["no.mr"]:
            in_chan -= 2

        super()._load_hyper_new_cnn(
            hyper=hyper,
            device_id=device_id,
            in_chan=in_chan,
            out_chan=out_chan,
        )

    def _load_hyper_loss_func(self, hyper: Dict, device_id: int):
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.get_device(device_id))

    def _load_hyper_data_sets(self, hyper: Dict):
        # load train/valid/test datasets
        for i in [DatasetPart.TRAIN, DatasetPart.VALID]:
            # only use data augmentation on training set
            if i == DatasetPart.TRAIN:
                augment = Dict()
                augment["augment.methods"] = hyper["augment.methods"]
                augment["augment.pct"] = hyper["augment.pct"]
                augment["augment.min"] = hyper["augment.min"]
                augment["augment.max"] = hyper["augment.max"]
            else:
                augment = None
            hyper["{}.set".format(i)] = BaselineDataSet(
                patients=hyper["{}.patients".format(i)],
                dataset_ver=hyper["dataset.ver"],
                no_pt=hyper["no.pt"],
                no_mr=hyper["no.mr"],
                augment=augment,
            )

    # baseline/idl.gtvn/idl.gtvs share this function
    def _load_hyper_data_loaders(self, hyper: Dict):
        for i in [DatasetPart.TRAIN, DatasetPart.VALID]:
            # only shuffle train loader
            if i == DatasetPart.TRAIN:
                shuffle = True
            else:
                shuffle = False
            hyper["{}.loader".format(i)] = DataLoader(
                dataset=hyper["{}.set".format(i)],
                batch_size=hyper["batch.size.actual"],
                shuffle=shuffle,
                num_workers=g.NUM_WORKERS,
            )

    def _simplify_hyper(self, hyper: Dict) -> Dict:
        # use simiple_hyper here, dont change origin hyper dict
        simple_hyper = super()._simplify_hyper(hyper)

        ignore_list = []
        for i in [
            DatasetPart.TRAIN,
            DatasetPart.VALID,
            DatasetPart.TEST,
        ]:
            ignore_list.append("{}.patients".format(i))
            ignore_list.append("{}.set".format(i))
            ignore_list.append("{}.loader".format(i))

        ignore_list.append("fold")

        # here in this for loop, use "hyper" instead of "simple_hyper"
        # otherwise will cause error: dictionary changed size during iteration
        for key_name in hyper.keys():
            if key_name in ignore_list:
                simple_hyper.pop(key_name)
            else:
                pass
        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self._simplify_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self._simplify_hyper(hyper)
        g.save_json(data=simple_hyper, path=json_path)

    # protected function, IDLGTVnTraining will inherit it
    def _plot_lr_fig(self, lr_json_path: str):
        plt.figure().clear()

        lr_dict = g.load_json(lr_json_path)
        lr_list = List()
        for i in lr_dict:
            lr_list.append(lr_dict[i])

        plt.plot(range(1, len(lr_list) + 1), lr_list, label="lr")
        plt.legend()
        plt.savefig(lr_json_path[:-4] + "png")

    # protected function, IDLGTVnTraining will inherit it
    def _plot_loss_fig(self, loss_json_path: str):
        loss_dict = g.load_json(loss_json_path)
        train_loss = List()
        valid_loss = List()

        for i in loss_dict:
            train_loss.append(loss_dict[i][DatasetPart.TRAIN])
            valid_loss.append(loss_dict[i][DatasetPart.VALID])

        # draw figure
        plt.figure().clear()
        plt.ylim(min(train_loss) - 0.05, max(train_loss) + 0.05)
        plt.plot(range(1, len(loss_dict) + 1), train_loss, label=DatasetPart.TRAIN)
        plt.plot(range(1, len(loss_dict) + 1), valid_loss, label=DatasetPart.VALID)
        plt.legend()
        plt.savefig(loss_json_path[:-4] + "png")

    def _calculate_loss(
        self,
        item: Dict,
        hyper: Dict,
        device_id: int,
    ):
        input_imgs = item["input.imgs"].to(g.get_device(device_id))
        labels = item["labels"].to(g.get_device(device_id))
        preds = hyper["cnn"](input_imgs)
        loss = hyper["loss.func"](preds, labels)
        return loss

    def _training_all_epochs(
        self,
        hyper: Dict,
        device_id: int,
        fold_dir: str,
    ):
        best_loss_dict = Dict()
        loss_json_path = os.path.join(fold_dir, "loss.json")
        lr_json_path = os.path.join(fold_dir, "lr.json")
        patience = 0
        scaler = GradScaler()

        for epoch in range(1, hyper["epochs"] + 1):
            print("")
            print("epoch: {}".format(epoch))
            print("training:")
            hyper["cnn"].train()
            train_loss = 0
            batch_count = 0
            hyper["train.loader"].dataset.set_epoch(epoch)
            hyper["valid.loader"].dataset.set_epoch(epoch)

            # Training loop
            for item in tqdm(hyper["train.loader"]):
                # Zero grad at the beginning of each mini-batch
                hyper["optim"].zero_grad()

                # Mixed precision training
                with autocast():
                    loss = self._calculate_loss(
                        item=item,
                        hyper=hyper,
                        device_id=device_id,
                    )

                # Backpropagation and optimization step
                # Get grad (must be after: optim.zero_grad())
                scaler.scale(loss).backward()
                # Update parameters
                scaler.step(hyper["optim"])
                scaler.update()

                # Accumulate training loss
                train_loss += loss.item()
                batch_count += 1

            # Calculate average training loss
            train_loss /= batch_count

            # validation
            print("validation:")
            valid_loss = 0
            batch_count = 0
            hyper["cnn"].eval()
            with torch.no_grad():
                for item in tqdm(hyper["valid.loader"]):
                    with autocast():
                        loss = self._calculate_loss(
                            item=item,
                            hyper=hyper,
                            device_id=device_id,
                        )

                    valid_loss += loss.item()
                    batch_count += 1

            valid_loss /= batch_count
            hyper["scheduler"].step(valid_loss)

            # current epoch finished
            hyper["epochs.actual"] = epoch

            # save loss in json
            loss_dict = g.load_json(loss_json_path)
            epoch_loss = Dict()
            epoch_loss[DatasetPart.TRAIN] = train_loss
            epoch_loss[DatasetPart.VALID] = valid_loss
            loss_dict["epoch={:03d}".format(hyper["epochs.actual"])] = epoch_loss
            g.save_json(loss_dict, loss_json_path)
            # draw loss figure
            self._plot_loss_fig(loss_json_path)

            # save lr in json
            lr_dict = g.load_json(lr_json_path)
            for param_group in hyper["optim"].param_groups:
                epoch_lr = param_group["lr"]
            lr_dict["epoch={:03d}".format(hyper["epochs.actual"])] = epoch_lr
            g.save_json(lr_dict, lr_json_path)
            # draw lr figure
            self._plot_lr_fig(lr_json_path)

            # save cnn
            if len(best_loss_dict) < hyper["keep.best.cnn.num"]:
                best_loss_dict[epoch] = valid_loss
                epoch_dir = os.path.join(fold_dir, "epoch={:03d}".format(epoch))
                g.create_dir(epoch_dir)
                self._save_cnn(
                    hyper=hyper,
                    save_path=os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch)),
                    device_id=device_id,
                )
            else:
                worst_epoch = best_loss_dict.key_with_max_value()
                worst_loss = best_loss_dict[worst_epoch]
                if valid_loss < worst_loss:
                    g.delete_path(
                        os.path.join(fold_dir, "epoch={:03d}".format(worst_epoch))
                    )
                    best_loss_dict.pop(worst_epoch)
                    best_loss_dict[epoch] = valid_loss
                    epoch_dir = os.path.join(fold_dir, "epoch={:03d}".format(epoch))
                    g.create_dir(epoch_dir)
                    self._save_cnn(
                        hyper=hyper,
                        save_path=os.path.join(
                            epoch_dir, "epoch={:03d}.pt".format(epoch)
                        ),
                        device_id=device_id,
                    )
                    patience = 0
                else:
                    patience += 1
                    if patience >= hyper["early.stop.epochs"]:
                        break

    def _training_all_folds(
        self,
        hyper: Dict,
        device_id: int,
        train_dir: str,
        idl_gtvn_baseline_id: str,
        debug_mode: bool,
    ):
        g.create_dir(train_dir)

        # check dataset version (before number of folds are comfirmed)
        self._load_hyper_dataset_version(
            hyper=hyper,
            idl_baseline_id=idl_gtvn_baseline_id,
        )

        # cross validation
        fold = int(hyper["fold"])
        fold = g.clamp_value(fold, (0, g.DATASET_FOLDS[hyper["dataset.ver"]]))
        # fold=0 will activate cross validation
        if fold == 0:
            fold_list = List(range(1, g.DATASET_FOLDS[hyper["dataset.ver"]] + 1))
        else:
            fold_list = [fold]

        # backup origin hyper for resetting hyper on next fold
        # (after "fold" removed from hyper Dict)
        hyper.pop("fold")
        origin_hyper = hyper.copy()

        # loop through each fold
        for fold in fold_list:
            fold_dir = os.path.join(train_dir, "fold={}".format(fold))
            g.create_dir(fold_dir)

            # load and print hyperparams
            self._load_hyper(
                hyper=hyper,
                device_id=device_id,
                idl_gtvn_baseline_id=idl_gtvn_baseline_id,
                fold=fold,
                debug_mode=debug_mode,
            )
            print("")
            self._print_hyper(hyper)

            print("")
            print("fold: {}".format(fold))

            # save an empty loss.json
            g.save_json(Dict(), os.path.join(fold_dir, "loss.json"))
            # save an empty lr.json
            g.save_json(Dict(), os.path.join(fold_dir, "lr.json"))

            # save hyper before training
            hyper_save_path = os.path.join(fold_dir, "hyper.json")
            self._save_hyper(hyper, hyper_save_path)

            # start training
            hyper["time.spent"] = datetime.now()
            self._training_all_epochs(
                hyper=hyper,
                device_id=device_id,
                fold_dir=fold_dir,
            )
            hyper["time.spent"] = datetime.now() - hyper["time.spent"]
            hyper["time.spent"] = str(hyper["time.spent"]).split(".", 2)[0]

            # save hyper after training
            self._save_hyper(hyper, hyper_save_path)

            # train at most 2 folds in debug mode
            if debug_mode and len(fold_list) > 1 and fold_list.index(fold) == 1:
                break

            # reset hyper before next fold
            hyper = origin_hyper.copy()

    def new_training(
        self,
        train_remark: str = "",
        device_id: int = -1,  # use all cards by default
        debug_mode: bool = False,
    ):
        self._new_training(
            idl_gtvn_baseline_id=None,
            train_remark=train_remark,
            device_id=device_id,
            debug_mode=debug_mode,
        )

    def _new_training(
        self,
        idl_gtvn_baseline_id: str = None,
        train_remark: str = "",
        device_id: int = -1,  # use all cards by default
        debug_mode: bool = False,
    ):
        if idl_gtvn_baseline_id is None:
            hyper_json_path = g.HYPER_PATH["baseline"]
        else:
            hyper_json_path = g.HYPER_PATH["idl.gtvn"]

        for hyper in self._load_hyper_series_from_json(hyper_json_path):
            # init train id
            if idl_gtvn_baseline_id is None:
                train_id = "baseline_"
            else:
                train_id = "idl.gtvn_"
            train_id += self._init_train_id(
                hyper=hyper,
                hyper_json_path=hyper_json_path,
                train_remark=train_remark,
                debug_mode=debug_mode,
            )
            print("")
            print(train_id)

            if idl_gtvn_baseline_id is None:
                train_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id, "baseline")
            else:
                train_dir = os.path.join(
                    g.TRAIN_RESULTS_DIR, idl_gtvn_baseline_id, train_id
                )

            self._training_all_folds(
                hyper=hyper,
                device_id=device_id,
                train_dir=train_dir,
                idl_gtvn_baseline_id=idl_gtvn_baseline_id,
                debug_mode=debug_mode,
            )

            # inference
            # (1) inference on all folds
            for dataset_part in [DatasetPart.VALID, DatasetPart.TEST]:
                self._inference_all_folds(
                    train_id=train_id,
                    dataset_ver=hyper["dataset.ver"],
                    dataset_part=dataset_part,
                    device_id=device_id,
                    debug_mode=debug_mode,
                )

            # (2) remove non-optimal epochs after inference
            self._remove_non_optimal_epochs(train_id)

            # (3) cross validation evaluation after non optimal epochs removed
            self._inference_cross_valid(
                train_id=train_id,
                device_id=device_id,
                dataset_ver=hyper["dataset.ver"],
                debug_mode=debug_mode,
            )

    def inference_all_folds(
        self,
        baseline_id: str,
        dataset_ver: str = None,
        dataset_part: str = DatasetPart.TEST,  # only valid or test
        device_id: int = 1,  # use card 1 by default
        debug_mode: bool = False,
    ):
        self._is_valid_baseline_id(baseline_id)
        self._inference_all_folds(
            train_id=baseline_id,
            dataset_ver=dataset_ver,
            dataset_part=dataset_part,
            device_id=device_id,
            debug_mode=debug_mode,
        )

    # this function is shared by baseline and idl.gtvn training
    def _inference_all_folds(
        self,
        train_id: str,
        dataset_ver: str = None,
        dataset_part: str = DatasetPart.TEST,  # only valid or test
        device_id: int = 1,  # use card 1 by default
        debug_mode: bool = False,
    ):
        print("")
        print("inference: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        if train_dir is None:
            g.error_exit("'train_id' not found!")

        fold_dirs = g.get_sub_dirs(train_dir, key_word="fold=", full_path=True)

        # load hyper json
        hyper = g.load_json(os.path.join(fold_dirs[0], "hyper.json"))

        # use geodesic.distance or not
        # (1) for baseline or for legacy hyper.json that doesn't have "geodesic.distance"
        if hyper["geodesic.distance"] == {}:
            geodesic_distance = False
            hyper.pop("geodesic.distance")
        # (2) for idl.gtvn
        else:
            geodesic_distance = bool(hyper["geodesic.distance"])

        # cnn trained with or without pt
        no_pt = hyper["no.pt"]
        if not no_pt and dataset_ver == DatasetVer.MDA:
            g.error_exit("inference on mda requires a cnn trained without pet.")

        # cnn trained with or without mr
        no_mr = hyper["no.mr"]
        if not no_mr and dataset_ver == DatasetVer.HECKTOR:
            g.error_exit("inference on hecktor requires a cnn trained without mr.")

        # dataset version
        training_dataset_ver = hyper["dataset.ver"]
        dataset_ver = self.__is_valid_inference_dataset_ver(
            inference_dataset_ver=dataset_ver,
            training_dataset_ver=training_dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))

        # dataset part
        # (1) inference on training dataset
        if dataset_ver == training_dataset_ver:
            if dataset_part not in [
                DatasetPart.VALID,
                DatasetPart.TEST,
            ]:
                g.error_exit(ErrMsg.DATASET_PART_INVALID)
        # (2) inference on other dataset
        else:
            if dataset_part != DatasetPart.TEST:
                g.error_exit(ErrMsg.DATASET_PART_INVALID)
        print("dataset part: {}".format(dataset_part))

        # load segmentation metrics
        if dataset_part == DatasetPart.VALID:
            metric_funcs = self._load_metric_funcs(device_id)
        # no need to load segmentation metrics
        else:
            metric_funcs = None

        # loop through fold dirs
        for fold_dir in fold_dirs:
            fold = int(Path(fold_dir).name[len("fold=") :])
            print("")
            print("fold: ", fold)

            # load patients
            patients = self._load_patients(
                dataset_ver=dataset_ver,
                fold=fold,
                debug_mode=debug_mode,
            )[dataset_part]

            # loop through epoch dirs
            for epoch_dir in g.get_sub_dirs(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("epoch: ", epoch)

                # load cnn
                cnn_path = os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch))
                cnn = self._load_exist_cnn(cnn_path, device_id=device_id)

                # initialize scores dict (valid set only)
                if dataset_part == DatasetPart.VALID:
                    epoch_scores = Dict()

                # loop through each patient
                for patient in tqdm(patients):
                    # (1)outputs structure for sigle-observer datasets:
                    # [gtv]->["dsc/msd/hd95/label/pred/clicks/distance.map"]

                    # (2)outputs structure for MDA dataset:
                    # [gtv]->["label/clicks/distance.map"]->["observer1/2/3"]
                    # [gtv]->["dsc/msd/hd95"]->["observer1/2/3/iov"]
                    patient_outputs = self._inference_single_patient(
                        patient=patient,
                        cnn=cnn,
                        dataset_ver=dataset_ver,
                        no_pt=no_pt,
                        no_mr=no_mr,
                        device_id=device_id,
                        metric_funcs=metric_funcs,
                        idl_gtvn_geodesic_distance=geodesic_distance,
                    )
                    if patient_outputs is None:
                        continue

                    # create folder and save preds of current patient
                    if dataset_part == DatasetPart.TEST:
                        self._inference_all_folds_save_patient_preds(
                            patient=patient,
                            epoch_dir=epoch_dir,
                            patient_outputs=patient_outputs,
                        )

                    # record score of current patient
                    # (on valid set only, the metrics are used to select the best epoch)
                    if dataset_part == DatasetPart.VALID:
                        self._inference_all_folds_record_patient_score(
                            patient=patient,
                            patient_outputs=patient_outputs,
                            scores=epoch_scores,
                        )

                # all patients under current epoch have been traversed
                # calculate median and avg score of current epoch
                if dataset_part == DatasetPart.VALID:
                    if len(epoch_scores) <= 0:
                        print("no label from current observer!")
                    else:
                        self._inference_calculate_avg_median_save_json(
                            scores=epoch_scores,
                            save_dir=epoch_dir,
                            dataset_ver=dataset_ver,
                            dataset_part=dataset_part,
                        )

                continue  # next epoch

    def _inference_all_folds_save_patient_preds(
        self,
        patient: str,
        epoch_dir: str,
        patient_outputs: Dict,
    ):
        patient_dir = os.path.join(
            epoch_dir,
            "patients",
            "patient={}".format(patient),
        )
        g.create_dir(patient_dir)

        for gtv in ["gtvt", "gtvn"]:
            g.save_nii(
                img=patient_outputs[gtv]["pred"],
                save_path=os.path.join(patient_dir, "{}_pred.nii.gz".format(gtv)),
                spacing=g.NII_SPACING,
            )

    def _inference_all_folds_record_patient_score(
        self,
        patient: str,
        patient_outputs: Dict,
        scores: Dict,
    ):
        for gtv in patient_outputs.keys():
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                # save cur patient metric
                scores["patient={}".format(patient)][gtv][metric] = patient_outputs[
                    gtv
                ][metric]
                # add cur patient metric into a list for avg and median calculation
                # initialize a list
                if scores[Stat.AVG][gtv][metric] == {}:
                    scores[Stat.AVG][gtv][metric] = List()
                # add current patient metric into the list
                scores[Stat.AVG][gtv][metric].append(patient_outputs[gtv][metric])

    def _inference_calculate_avg_median_save_json(
        self,
        scores: Dict,
        save_dir: str,
        dataset_ver: str,
        dataset_part: str,
        mda_obs: str = None,
    ):
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores[Stat.MEDIAN][gtv][metric] = g.calculate_median(
                    scores[Stat.AVG][gtv][metric]
                )
                scores[Stat.AVG][gtv][metric] = g.calculate_avg(
                    scores[Stat.AVG][gtv][metric]
                )

        # save scores in json
        if mda_obs is None:
            json_name = "inference_{}_{}.json".format(dataset_ver, dataset_part)
        else:
            json_name = "inference_{}_{}_{}.json".format(
                dataset_ver, dataset_part, mda_obs
            )
        g.save_json(
            data=scores,
            path=os.path.join(save_dir, json_name),
        )

    def inference_cross_valid(
        self,
        baseline_id: str,
        dataset_ver: str = None,  # au/mda
        mda_obs: str = None,
        device_id: int = 1,  # use card 1 by default
        debug_mode: bool = False,
    ):
        self._is_valid_baseline_id(baseline_id)
        self._inference_cross_valid(
            train_id=baseline_id,
            dataset_ver=dataset_ver,
            mda_obs=mda_obs,
            device_id=device_id,
            debug_mode=debug_mode,
        )

    def _inference_cross_valid_init_scores(self, *args, **kwargs):
        scores = Dict()
        return scores

    def _inference_cross_valid(
        self,
        train_id: str,
        dataset_ver: str = None,
        mda_obs: str = None,
        device_id: int = 1,  # use card 1 by default
        debug_mode: bool = False,
    ):
        print("")
        print("cross valid evaluation: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        if train_dir is None:
            g.error_exit("'train_id' not found!")

        fold_dirs = g.get_sub_dirs(train_dir, key_word="fold=", full_path=True)

        hyper = g.load_json(os.path.join(fold_dirs[0], "hyper.json"))

        # dataset ver
        training_dataset_ver = hyper["dataset.ver"]
        dataset_ver = self.__is_valid_inference_dataset_ver(
            inference_dataset_ver=dataset_ver,
            training_dataset_ver=training_dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))

        # load segmentation metrics
        metric_funcs = self._load_metric_funcs(device_id)

        # create folder in train_dir to save cross_valid preds
        g.create_dir(os.path.join(Path(fold_dirs[0]).parent, "patients"))

        patients = self._load_patients(
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )[DatasetPart.TEST]

        # mda observer filter
        if dataset_ver != DatasetVer.MDA:
            mda_obs = None
        if mda_obs is not None:
            for patient in patients.copy():
                if mda_obs not in patient:
                    patients.remove(patient)

        # initialize scores dict
        scores = self._inference_cross_valid_init_scores(
            idl_gtvn_dir=train_dir,
            dataset_ver=dataset_ver,
            patients=patients,
        )

        for patient in tqdm(patients):
            # initialize preds
            preds = Dict()
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                preds[gtv] = None

            for fold_dir in fold_dirs:
                # find epoch dir
                epoch_dirs = g.get_sub_dirs(fold_dir, key_word="epoch=", full_path=True)
                if len(epoch_dirs) > 1:
                    self._remove_non_optimal_epochs(train_id)
                    epoch_dir = g.get_sub_dirs(
                        fold_dir, key_word="epoch=", full_path=True
                    )[0]
                else:
                    epoch_dir = epoch_dirs[0]

                # load preds
                patient_dir = os.path.join(
                    epoch_dir, "patients", "patient={}".format(patient)
                )
                for gtv in ["gtvt", "gtvn"]:
                    pred_path = os.path.join(patient_dir, "{}_pred.nii.gz".format(gtv))
                    if os.path.exists(pred_path):
                        img = g.load_nii(path=pred_path, binary=False)
                        if preds[gtv] is None:
                            preds[gtv] = img
                        else:
                            preds[gtv] += img

            # all folds is traversed
            # for idl.gtvn pred["gtvt"] will be None
            if preds["gtvt"] is not None:
                preds["gtvs"] = preds["gtvt"] + preds["gtvn"]

            # calculate avg pred value over all folds
            for gtv in preds.keys():
                if preds[gtv] is None:
                    preds.pop(gtv)
                else:
                    preds[gtv] /= len(fold_dirs)

            # create cross_valid dir
            pred_dir = os.path.join(
                Path(fold_dirs[0]).parent, "patients", "patient={}".format(patient)
            )
            # preds.keys() = ["gtvt", "gtvn", "gtvs"] for baseline
            # preds.keys() = ["gtvn"] for idl.gtvn
            if len(preds.keys()) == 1:
                pred_dir = os.path.join(pred_dir, "round=01")
            g.create_dir(pred_dir)

            # save cross_valid preds (only save gtvt and gtvn)
            for gtv in preds.keys():
                if gtv != "gtvs":
                    g.save_nii(
                        img=preds[gtv],
                        save_path=os.path.join(pred_dir, "{}_pred.nii.gz".format(gtv)),
                        spacing=g.NII_SPACING,
                    )

            # calculate metrics
            labels = g.load_gtv_labels(
                dataset_ver=dataset_ver,
                patient=patient,
            )

            # calculate metrics
            self._inference_cross_valid_record_patient_score(
                patient=patient,
                preds=preds,
                labels=labels,
                metric_funcs=metric_funcs,
                scores=scores,
            )

        # all patients have been traversed
        # calculate avg and median score
        self._inference_calculate_avg_median_save_json(
            scores=scores,
            save_dir=Path(fold_dirs[0]).parent,
            dataset_ver=dataset_ver,
            dataset_part=DatasetPart.TEST,
            mda_obs=mda_obs,
        )

    def _inference_cross_valid_record_patient_score(
        self,
        patient: str,
        preds: Dict,
        labels: Dict,
        metric_funcs: Dict,
        scores: Dict,
    ):
        for gtv in preds.keys():
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                score = metric_funcs[metric](preds[gtv], labels[gtv])

                # save cur patient metric
                scores["patient={}".format(patient)][gtv][metric] = score

                # add cur patient metric into a list for avg and median calculation
                if scores[Stat.AVG][gtv][metric] == {}:
                    scores[Stat.AVG][gtv][metric] = List()
                scores[Stat.AVG][gtv][metric].append(score)

    def _remove_non_optimal_epochs(self, train_id: str):
        print("")
        print("remove non optimal epochs: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        fold_dirs = g.get_sub_dirs(train_dir, key_word="fold=", full_path=True)

        # load dataset version
        dataset_ver = g.load_json(os.path.join(fold_dirs[0], "hyper.json"))[
            "dataset.ver"
        ]
        self._is_valid_dataset_ver(dataset_ver)

        # this "inference_{}_valid.json" file:
        # (1) is created by "inference_all_folds()"
        # (2) only saves validation metrics
        inference_json_name = "inference_{}_valid.json".format(dataset_ver)

        for fold_dir in fold_dirs:
            fold_scores = Dict()

            for epoch_dir in g.get_sub_dirs(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                # load and record scores of current epoch
                epoch_scores = g.load_json(os.path.join(epoch_dir, inference_json_name))

                self._remove_non_optimal_epochs_record_epoch_scores(
                    fold_scores=fold_scores,
                    epoch_scores=epoch_scores,
                    epoch=epoch,
                )

            best_epoch = self._remove_non_optimal_epochs_find_best_epoch(
                scores=fold_scores
            )

            # delete non-optimal epochs
            for epoch_dir in g.get_sub_dirs(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                if epoch != best_epoch:
                    g.delete_path(epoch_dir)
                    print("delete: {} {}".format(Path(fold_dir).name, epoch))

    def _remove_non_optimal_epochs_record_epoch_scores(
        self,
        fold_scores: Dict,
        epoch_scores: Dict,
        epoch: str,
    ):
        for stat in [Stat.MEDIAN, Stat.AVG]:
            fold_scores[epoch][stat] = epoch_scores[stat]

    # a sub function of _remove_non_optimal_epochs()
    def _remove_non_optimal_epochs_find_best_epoch(
        self, scores: Dict, gtv_list: list = ["gtvs", "gtvt", "gtvn"]
    ):
        for stat in [Stat.MEDIAN, Stat.AVG]:
            for gtv in gtv_list:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    # create a tmp list to sort
                    list_to_sort = List()

                    # add elements into the list
                    for epoch in scores.keys():
                        if len(gtv_list) > 1:
                            cur_score = scores[epoch][stat][gtv][metric]
                        else:
                            cur_score = scores[epoch][stat][metric]
                        list_to_sort.append(cur_score)

                    # sort the list
                    if metric == Metric.DSC:
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)

                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        if len(gtv_list) > 1:
                            cur_score = scores[epoch][stat][gtv][metric]
                        else:
                            cur_score = scores[epoch][stat][metric]

                        new_value = list_to_sort.index(cur_score)
                        if metric == Metric.DSC:
                            new_value *= 2

                        if len(gtv_list) > 1:
                            scores[epoch][stat][gtv][metric] = new_value
                        else:
                            scores[epoch][stat][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stat in [Stat.AVG, Stat.MEDIAN]:
                for gtv in gtv_list:
                    for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                        if len(gtv_list) > 1:
                            evaluation[epoch] += scores[epoch][stat][gtv][metric]
                        else:
                            evaluation[epoch] += scores[epoch][stat][metric]

        return evaluation.key_with_max_value()

    def _inference_single_patient_record_labels(
        self,
        outputs: Dict,
        dataset_item: Dict,
    ):
        img_shape = dataset_item["shape"]

        labels = Dict()
        labels["gtvt"] = dataset_item["labels"][1].cpu().numpy()
        labels["gtvn"] = dataset_item["labels"][2].cpu().numpy()

        for gtv in ["gtvt", "gtvn"]:
            labels[gtv] = g.center_align_img(labels[gtv], img_shape)

        labels["gtvs"] = np.maximum(labels["gtvt"], labels["gtvn"])

        for gtv in ["gtvt", "gtvn", "gtvs"]:
            outputs[gtv]["label"] = labels[gtv]

        return outputs

    def _inference_single_patient_record_preds(
        self,
        outputs: Dict,
        preds: ndarray,
        img_shape: tuple,
    ):
        # preds: [background, gtvt, gtvn]
        outputs["gtvt"]["pred"] = g.center_align_img(preds[1], img_shape)
        outputs["gtvn"]["pred"] = g.center_align_img(preds[2], img_shape)
        outputs["gtvs"]["pred"] = np.maximum(
            outputs["gtvt"]["pred"], outputs["gtvn"]["pred"]
        )

    def __is_valid_inference_dataset_ver(
        self,
        inference_dataset_ver: str,
        training_dataset_ver: str,
    ):
        if training_dataset_ver not in [
            DatasetVer.AU,
            DatasetVer.MDA,
            DatasetVer.NKI,
            DatasetVer.HECKTOR,
        ]:
            g.error_exit(ErrMsg.DATASET_VER_INVALID)

        # copy from training_dataset_ver if inference_dataset_ver is None
        if inference_dataset_ver is None:
            inference_dataset_ver = training_dataset_ver

        # on AU dataset
        if training_dataset_ver == DatasetVer.AU:
            self._is_valid_dataset_ver(inference_dataset_ver)
        # on other dataset
        else:
            # make sure inference_dataset_ver = training_dataset_ver
            if inference_dataset_ver != training_dataset_ver:
                g.error_exit(ErrMsg.DATASET_VER_INVALID)

        return inference_dataset_ver
