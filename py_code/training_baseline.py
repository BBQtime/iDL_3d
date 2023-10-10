import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from custom import GPU, DatasetPart, DatasetVer, Debug, Dict, DirExplorer, Folder
from custom import Global as g
from custom import Img, Json, List, Metric, Nii, Value
from dataset_baseline import DataSetBaseline
from loss_func import UnifiedFocalLoss
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_core import TrainingCore


class TrainingBaseline(TrainingCore):
    def _load_hyper(
        self, hyper: Dict, fold: int, idl_gtvn_baseline_id: str, debug_mode: bool
    ):
        # shared by baseline/idl.gtvn/idl.gtvt
        super()._load_hyper(hyper)

        # epochs
        if debug_mode:
            # only train 2 epoch in debug mode
            hyper["epochs"] = 2
        else:
            hyper["epochs"] = Value.limit_range(hyper["epochs"], (1, None))

        # record actual epochs because of early stop
        hyper["epochs.actual"] = 0

        # early stop, based on epoch
        hyper["early.stop.epochs"] = Value.limit_range(
            hyper["early.stop.epochs"], (1, hyper["epochs"])
        )

        # lr
        hyper["lr"] = Value.limit_range(hyper["lr"], (g.EPS, 1.0))

        # actual lr
        if GPU.used_count() > 1:
            hyper["lr.actual"] = hyper["lr"] * GPU.used_count()
        else:
            hyper["lr.actual"] = hyper["lr"]

        # min lr
        hyper["lr.min"] = Value.limit_range(hyper["lr.min"], (g.EPS, hyper["lr"]))

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        hyper["lr.decay.patience"] = Value.limit_range(
            hyper["lr.decay.patience"], (1, hyper["epochs"])
        )

        # number of best valid loss cnn retained
        hyper["keep.best.cnn.num"] = Value.limit_range(
            hyper["keep.best.cnn.num"], (1, hyper["epochs"])
        )

        # augment percent
        hyper["augment.pct"] = Value.limit_range(hyper["augment.pct"], (0.0, 1.0))

        self._load_hyper_dataset_version(
            hyper=hyper,
            idl_baseline_id=idl_gtvn_baseline_id,
        )

        # load patients after dataset version is selected
        patients = self._load_patients(
            dataset_ver=hyper["dataset.ver"],
            fold=fold,
            debug_mode=debug_mode,
        )
        for key_name in [DatasetPart.TRAIN, DatasetPart.VALID]:
            hyper["{}.patients".format(key_name)] = patients[key_name]

        self._load_hyper_loss_func(hyper)

        # load datasets before load dataloaders
        self._load_hyper_data_sets(
            hyper=hyper, idl_gtvn_baseline_id=idl_gtvn_baseline_id
        )

        self._load_hyper_data_loaders(hyper)

        # load cnn before optimizer
        self._load_hyper_new_cnn(hyper=hyper)

        self._load_hyper_optim_and_scheduler(hyper=hyper)

    def _load_hyper_new_cnn(self, hyper: Dict, in_chan: int = 4, out_chan: int = 3):
        if hyper["no.pt"]:
            in_chan -= 1
        super()._load_hyper_new_cnn(hyper=hyper, in_chan=in_chan, out_chan=out_chan)

    def _load_hyper_loss_func(self, hyper: Dict):
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

    def _load_hyper_data_sets(self, hyper: Dict, idl_gtvn_baseline_id: str = None):
        # load train/valid/test datasets
        for i in [DatasetPart.TRAIN, DatasetPart.VALID]:
            # only use data augmentation on training set
            if i == DatasetPart.TRAIN:
                augment = Dict()
                augment["methods"] = hyper["augment.methods"]
                augment["pct"] = hyper["augment.pct"]
                augment["min"] = hyper["augment.min"]
                augment["max"] = hyper["augment.max"]
            else:
                augment = None
            hyper["{}.set".format(i)] = DataSetBaseline(
                patients=hyper["{}.patients".format(i)],
                dataset_ver=hyper["dataset.ver"],
                no_pt=hyper["no.pt"],
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
            DatasetPart.TEST_INTER,
            DatasetPart.TEST_EXTER,
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
        Json.save(data=simple_hyper, path=json_path)

    # protected function, TrainingIDLGTVn will inherit it
    def _plot_lr_fig(self, lr_json_path: str):
        plt.figure().clear()

        lr_dict = Json.load(lr_json_path)
        lr_list = List()
        for i in lr_dict:
            lr_list.append(lr_dict[i])

        plt.plot(range(1, len(lr_list) + 1), lr_list, label="lr")
        plt.legend()
        plt.savefig(lr_json_path[:-4] + "png")

    # protected function, TrainingIDLGTVn will inherit it
    def _plot_loss_fig(self, loss_json_path: str):
        loss_dict = Json.load(loss_json_path)
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

    def _calculate_loss(self, item: tuple, hyper: Dict):
        input_imgs = item[0].to(g.DEVICE)
        labels = item[1].to(g.DEVICE)
        preds = hyper["cnn"](input_imgs)
        loss = hyper["loss.func"](preds, labels)
        return loss

    def _training_all_epochs(self, hyper: Dict, fold_dir: str):
        best_loss_dict = Dict()
        loss_json_path = os.path.join(fold_dir, "loss.json")
        lr_json_path = os.path.join(fold_dir, "lr.json")
        patience = 0

        for epoch in range(1, hyper["epochs"] + 1):
            print("")
            print("epoch: {}".format(epoch))
            print("training:")
            hyper["cnn"].train()
            train_loss = 0
            batch_count = 0

            # training
            for item in tqdm(hyper["train.loader"]):
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                loss = self._calculate_loss(item=item, hyper=hyper)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                train_loss += loss.item()
                batch_count += 1
            train_loss /= batch_count

            # validation
            print("validation:")
            valid_loss = 0
            batch_count = 0
            hyper["cnn"].eval()
            with torch.no_grad():
                for item in tqdm(hyper["valid.loader"]):
                    loss = self._calculate_loss(item=item, hyper=hyper)
                    valid_loss += loss.item()
                    batch_count += 1
            valid_loss /= batch_count
            hyper["scheduler"].step(valid_loss)

            # current epoch finished
            hyper["epochs.actual"] = epoch

            # save loss in json
            loss_dict = Json.load(loss_json_path)
            epoch_loss = Dict()
            epoch_loss[DatasetPart.TRAIN] = train_loss
            epoch_loss[DatasetPart.VALID] = valid_loss
            loss_dict["epoch={:03d}".format(hyper["epochs.actual"])] = epoch_loss
            Json.save(loss_dict, loss_json_path)
            # draw loss figure
            self._plot_loss_fig(loss_json_path)

            # save lr in json
            lr_dict = Json.load(lr_json_path)
            for param_group in hyper["optim"].param_groups:
                epoch_lr = param_group["lr"]
            lr_dict["epoch={:03d}".format(hyper["epochs.actual"])] = epoch_lr
            Json.save(lr_dict, lr_json_path)
            # draw lr figure
            self._plot_lr_fig(lr_json_path)

            # save cnn
            if len(best_loss_dict) < hyper["keep.best.cnn.num"]:
                best_loss_dict[epoch] = valid_loss
                epoch_dir = os.path.join(fold_dir, "epoch={:03d}".format(epoch))
                Folder.create(epoch_dir)
                self._save_cnn(
                    hyper,
                    os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch)),
                )
            else:
                worst_epoch = best_loss_dict.key_with_max_value()
                worst_loss = best_loss_dict[worst_epoch]
                if valid_loss < worst_loss:
                    Folder.delete(
                        os.path.join(fold_dir, "epoch={:03d}".format(worst_epoch))
                    )
                    best_loss_dict.pop(worst_epoch)
                    best_loss_dict[epoch] = valid_loss
                    epoch_dir = os.path.join(fold_dir, "epoch={:03d}".format(epoch))
                    Folder.create(epoch_dir)
                    self._save_cnn(
                        hyper,
                        os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch)),
                    )
                    patience = 0
                else:
                    patience += 1
                    if patience >= hyper["early.stop.epochs"]:
                        break

    def _training_all_folds(
        self, hyper: Dict, train_dir: str, idl_gtvn_baseline_id: str, debug_mode: bool
    ):
        Folder.create(train_dir)

        # cross validation
        fold = int(hyper["fold"])
        fold = Value.limit_range(fold, (0, g.DATASET_FOLDS))
        # fold=0 will activate cross validation
        if fold == 0:
            fold_list = List(range(1, g.DATASET_FOLDS + 1))
        else:
            fold_list = [fold]

        # backup origin hyper for resetting hyper on next fold
        # (after "fold" removed from hyper Dict)
        hyper.pop("fold")
        origin_hyper = hyper.copy()

        # loop through each fold
        for fold in fold_list:
            fold_dir = os.path.join(train_dir, "fold={}".format(fold))
            Folder.create(fold_dir)

            # load and print hyperparams
            self._load_hyper(
                hyper=hyper,
                fold=fold,
                idl_gtvn_baseline_id=idl_gtvn_baseline_id,
                debug_mode=debug_mode,
            )
            print("")
            self._print_hyper(hyper)

            print("")
            print("fold: {}".format(fold))

            # save an empty loss.json
            Json.save(Dict(), os.path.join(fold_dir, "loss.json"))
            # save an empty lr.json
            Json.save(Dict(), os.path.join(fold_dir, "lr.json"))

            # save hyper before training
            hyper_save_path = os.path.join(fold_dir, "hyper.json")
            self._save_hyper(hyper, hyper_save_path)

            # start training
            hyper["time.spent"] = datetime.now()
            self._training_all_epochs(hyper, fold_dir)
            hyper["time.spent"] = datetime.now() - hyper["time.spent"]
            hyper["time.spent"] = str(hyper["time.spent"]).split(".", 2)[0]

            # save hyper after training
            self._save_hyper(hyper, hyper_save_path)

            # only train 2 folds in debug mode
            if (
                debug_mode
                and len(fold_list) > 1
                and fold_list.index(hyper["fold"]) == 1
            ):
                break

            # reset hyper before next fold
            hyper = origin_hyper.copy()

    def new_training(
        self,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        self._new_training(
            idl_gtvn_baseline_id=None,
            train_remark=train_remark,
            debug_mode=debug_mode,
        )

    def _new_training(
        self,
        idl_gtvn_baseline_id: str = None,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        if idl_gtvn_baseline_id is None:
            hyper_json_path = g.HYPER_JSON_PATH["baseline"]
        else:
            hyper_json_path = g.HYPER_JSON_PATH["idl.gtvn"]

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
                train_dir=train_dir,
                idl_gtvn_baseline_id=idl_gtvn_baseline_id,
                debug_mode=debug_mode,
            )

            # inference
            if hyper["dataset.ver"] == DatasetVer.MDA:
                dataset_part_list = [DatasetPart.VALID, DatasetPart.TEST]
            else:
                dataset_part_list = [DatasetPart.VALID, DatasetPart.TEST_INTER]

            # if baseline, save pred of training set
            if idl_gtvn_baseline_id is None:
                dataset_part_list.append(DatasetPart.TRAIN)

            for dataset_part in dataset_part_list:
                self._fold_wise_inference(
                    train_id=train_id,
                    dataset_ver=hyper["dataset.ver"],
                    dataset_part=dataset_part,
                    debug_mode=debug_mode,
                )

            # remove non-optimal epochs after inference
            self._remove_non_optimal_epochs(train_id)

            # cross validation evaluation after non optimal epochs removed
            dataset_part_list.remove(DatasetPart.VALID)
            for dataset_part in dataset_part_list:
                self._cross_valid_inference(
                    train_id=train_id,
                    dataset_ver=hyper["dataset.ver"],
                    dataset_part=dataset_part,
                    debug_mode=debug_mode,
                )

    def fold_wise_inference(
        self,
        baseline_id: str,
        dataset_part: str,  # train/valid/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self._is_valid_baseline_id(baseline_id)
        self._fold_wise_inference(
            train_id=baseline_id,
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _fold_wise_inference(
        self,
        train_id: str,
        dataset_part: str,  # train/valid/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        print("")
        print("inference: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        if train_dir is None:
            Debug.error_exit("training id not found")

        baseline_id = Path(train_dir).parent.name

        fold_dirs = DirExplorer.get_sub_folders(
            train_dir, key_word="fold=", full_path=True
        )

        hyper = Json.load(os.path.join(fold_dirs[0], "hyper.json"))
        no_pt = hyper["no.pt"]
        training_dataset_ver = hyper["dataset.ver"]

        dataset_ver = self._is_valid_dataset_version(
            dataset_ver=dataset_ver,
            origin_dataset_ver=training_dataset_ver,
        )
        self._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))
        print("dataset section: {}".format(dataset_part))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics(dataset_ver)

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
            )

            # loop through epoch dirs
            for epoch_dir in DirExplorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("epoch: ", epoch)

                # load cnn
                cnn_path = os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch))
                cnn = self._load_exist_cnn(cnn_path)

                # initialize scores dict (only for test sets)
                if (
                    DatasetPart.TEST in dataset_part
                    or DatasetPart.VALID in dataset_part
                ):
                    epoch_scores = self._inference_init_scores(
                        baseline_id=baseline_id,
                        dataset_ver=dataset_ver,
                        dataset_part=dataset_part,
                        patients=patients,
                    )

                # loop through each patient
                for patient in tqdm(patients[dataset_part]):
                    # outputs structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_outputs = self._inference_single_patient(
                        patient=patient,
                        cnn=cnn,
                        dataset_ver=dataset_ver,
                        dataset_part=dataset_part,
                        no_pt=no_pt,
                        segment_metrics=segment_metrics,
                        idl_gtvn_baseline_id=baseline_id,
                    )

                    # create folder and save preds of current patient
                    self._fold_wise_inference_save_patient_preds(
                        patient=patient,
                        epoch_dir=epoch_dir,
                        patient_outputs=patient_outputs,
                        dataset_ver=dataset_ver,
                        dataset_part=dataset_part,
                    )

                    # record score of current patient (test and valid sets only)
                    if (
                        DatasetPart.TEST in dataset_part
                        or DatasetPart.VALID in dataset_part
                    ):
                        self._fold_wise_inference_record_patient_score(
                            patient=patient,
                            patient_outputs=patient_outputs,
                            scores=epoch_scores,
                        )

                # all patients under current epoch have been traversed
                # calculate median and avg score of current epoch
                if (
                    DatasetPart.TEST in dataset_part
                    or DatasetPart.VALID in dataset_part
                ):
                    self._inference_calculate_save_avg_median(
                        scores=epoch_scores,
                        save_dir=epoch_dir,
                        dataset_ver=dataset_ver,
                        dataset_part=dataset_part,
                    )

                continue  # next epoch

    def _inference_init_scores(
        self,
        baseline_id: str = None,
        dataset_ver: str = None,
        dataset_part: str = None,
        patients: Dict = None,
    ) -> Dict:
        scores = Dict()
        for stats in ["median", "avg"]:
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    scores[stats][gtv][metric] = List()

        return scores

    def _fold_wise_inference_save_patient_preds(
        self,
        patient: str,
        epoch_dir: str,
        patient_outputs: Dict,
        dataset_ver: str,
        dataset_part: str,
    ):
        patient_dir = os.path.join(
            epoch_dir,
            "patients",
            "patient={}".format(patient),
        )
        Folder.create(patient_dir)

        for gtv in ["gtvt", "gtvn"]:
            Nii.save(
                img=patient_outputs[gtv]["pred"],
                save_path=os.path.join(patient_dir, "{}_pred.nii".format(gtv)),
                spacing=g.NII_SPACING[dataset_ver],
            )

    def _fold_wise_inference_record_patient_score(
        self, patient: str, patient_outputs: Dict, scores: Dict
    ):
        for gtv in patient_outputs.keys():
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                # save cur patient score
                scores["patient={}".format(patient)][gtv][metric] = patient_outputs[
                    gtv
                ][metric]
                # add scores of current patient into avg and median
                for stats in ["median", "avg"]:
                    scores[stats][gtv][metric].append(patient_outputs[gtv][metric])

    def _inference_calculate_save_avg_median(
        self, scores: Dict, save_dir: str, dataset_ver: str, dataset_part: str
    ):
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores["median"][gtv][metric] = Value.median(
                    scores["median"][gtv][metric]
                )
                scores["avg"][gtv][metric] = Value.avg(scores["avg"][gtv][metric])

        # save scores in json
        Json.save(
            data=scores,
            path=os.path.join(
                save_dir, "inference_{}_{}.json".format(dataset_ver, dataset_part)
            ),
        )

    def cross_valid_inference(
        self,
        baseline_id: str,
        dataset_part: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self._is_valid_baseline_id(baseline_id)
        self._cross_valid_inference(
            train_id=baseline_id,
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _cross_valid_inference(
        self,
        train_id: str,
        dataset_part: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        print("")
        print("cross valid evaluation: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        if train_dir is None:
            Debug.error_exit("training id not found")

        baseline_id = Path(train_dir).parent.name

        fold_dirs = DirExplorer.get_sub_folders(
            train_dir, key_word="fold=", full_path=True
        )

        hyper = Json.load(os.path.join(fold_dirs[0], "hyper.json"))
        training_dataset_ver = hyper["dataset.ver"]

        dataset_ver = self._is_valid_dataset_version(
            dataset_ver=dataset_ver,
            origin_dataset_ver=training_dataset_ver,
        )
        self._cross_valid_inference_is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))
        print("dataset section: {}".format(dataset_part))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics(dataset_ver)

        # create folder in train_dir to save cross_valid preds
        Folder.create(os.path.join(Path(fold_dirs[0]).parent, "patients"))

        patients = self._load_patients(
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

        # initialize scores dict
        if DatasetPart.TEST in dataset_part:
            scores = self._inference_init_scores(
                baseline_id=baseline_id,
                dataset_ver=dataset_ver,
                dataset_part=dataset_part,
                patients=patients,
            )

        for patient in tqdm(patients[dataset_part]):
            # initialize preds
            preds = Dict()
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                preds[gtv] = None

            for fold_dir in fold_dirs:
                # find epoch dir
                epoch_dirs = DirExplorer.get_sub_folders(
                    fold_dir, key_word="epoch=", full_path=True
                )
                if len(epoch_dirs) > 1:
                    self.remove_non_optimal_epochs(train_id)
                    epoch_dir = DirExplorer.get_sub_folders(
                        fold_dir, key_word="epoch=", full_path=True
                    )[0]
                else:
                    epoch_dir = epoch_dirs[0]

                # load preds
                patient_dir = os.path.join(
                    epoch_dir, "patients", "patient={}".format(patient)
                )
                for gtv in ["gtvt", "gtvn"]:
                    pred_path = os.path.join(patient_dir, "{}_pred.nii".format(gtv))
                    if os.path.exists(pred_path):
                        img = Nii.load(path=pred_path, binary=False)
                        if preds[gtv] is None:
                            preds[gtv] = img
                        else:
                            preds[gtv] += img

            # all folds is traversed
            # for idl.gtvn pred["gtvt"] will be None
            if preds["gtvt"] is not None:
                preds["gtvs"] = preds["gtvt"] + preds["gtvn"]

            for gtv in preds.keys():
                if preds[gtv] is None:
                    preds.pop(gtv)
                else:
                    preds[gtv] /= len(fold_dirs)

            # create cross_valid dir
            pred_dir = os.path.join(
                Path(fold_dirs[0]).parent, "patients", "patient={}".format(patient)
            )
            if len(preds.keys()) == 1:
                pred_dir = os.path.join(pred_dir, "round=01")
            Folder.create(pred_dir)

            # save cross_valid preds (only save gtvt and gtvn)
            for gtv in preds.keys():
                if gtv != "gtvs":
                    Nii.save(
                        img=preds[gtv],
                        save_path=os.path.join(pred_dir, "{}_pred.nii".format(gtv)),
                        spacing=g.NII_SPACING[dataset_ver],
                    )

            # load labels and calculate metrics (on test set only)
            if DatasetPart.TEST in dataset_part:
                labels = Img.load_labels(
                    dataset_dir=g.DATASET_DIR[dataset_ver], patient=patient
                )
                self._cross_valid_inference_record_patient_score(
                    patient=patient,
                    preds=preds,
                    labels=labels,
                    segment_metrics=segment_metrics,
                    scores=scores,
                )

        # all patients have been traversed
        # calculate avg and median score (on test set only)
        if DatasetPart.TEST in dataset_part:
            self._inference_calculate_save_avg_median(
                scores=scores,
                save_dir=Path(fold_dirs[0]).parent,
                dataset_part=dataset_part,
                dataset_ver=dataset_ver,
            )

    def _cross_valid_inference_is_valid_dataset_part(
        self,
        dataset_part: str,
        dataset_ver: str,
    ):
        if DatasetPart.VALID in dataset_part:
            Debug.error_exit("set dataset_part to 'train' instead of 'valid'")

        self._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )

    def _cross_valid_inference_record_patient_score(
        self,
        patient: str,
        preds: Dict,
        labels: Dict,
        segment_metrics: Dict,
        scores: Dict,
    ):
        for gtv in preds.keys():
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                score = segment_metrics[metric](preds[gtv], labels[gtv])
                # record current score
                scores["patient={}".format(patient)][gtv][metric] = score
                # record scores for avg and median score calculation
                for stats in ["median", "avg"]:
                    scores[stats][gtv][metric].append(score)

    def remove_non_optimal_epochs(self, baseline_id: str):
        self._is_valid_baseline_id(baseline_id)
        self._remove_non_optimal_epochs(baseline_id)

    def _remove_non_optimal_epochs(self, train_id: str):
        print("")
        print("remove non optimal epochs: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        fold_dirs = DirExplorer.get_sub_folders(
            train_dir, key_word="fold=", full_path=True
        )

        # load dataset version
        dataset_ver = Json.load(os.path.join(fold_dirs[0], "hyper.json"))["dataset.ver"]
        dataset_ver = self._is_valid_dataset_version(dataset_ver=dataset_ver)

        inference_json_name = "inference_{}_valid.json".format(dataset_ver)

        for fold_dir in fold_dirs:
            fold_scores = Dict()

            for epoch_dir in DirExplorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                # load and record scores of current epoch
                epoch_scores = Json.load(os.path.join(epoch_dir, inference_json_name))

                self._remove_non_optimal_epochs_record_epoch_scores(
                    fold_scores=fold_scores, epoch_scores=epoch_scores, epoch=epoch
                )

            best_epoch = self._remove_non_optimal_epochs_find_best_epoch(
                scores=fold_scores
            )

            # delete non-optimal epochs
            for epoch_dir in DirExplorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                if epoch != best_epoch:
                    Folder.delete(epoch_dir)
                    print("delete: {} {}".format(Path(fold_dir).name, epoch))

    def _remove_non_optimal_epochs_record_epoch_scores(
        self, fold_scores: Dict, epoch_scores: Dict, epoch: str
    ):
        for stats in ["median", "avg"]:
            fold_scores[epoch][stats] = epoch_scores[stats]

    # a sub function of _remove_non_optimal_epochs()
    def _remove_non_optimal_epochs_find_best_epoch(
        self, scores: Dict, gtv_list: list = ["gtvs", "gtvt", "gtvn"]
    ):
        for stats in ["median", "avg"]:
            for gtv in gtv_list:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    # create a tmp list to sort
                    list_to_sort = List()

                    # add elements into the list
                    for epoch in scores.keys():
                        if len(gtv_list) > 1:
                            cur_score = scores[epoch][stats][gtv][metric]
                        else:
                            cur_score = scores[epoch][stats][metric]
                        list_to_sort.append(cur_score)

                    # sort the list
                    if metric == Metric.DSC:
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)

                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        if len(gtv_list) > 1:
                            cur_score = scores[epoch][stats][gtv][metric]
                        else:
                            cur_score = scores[epoch][stats][metric]

                        new_value = list_to_sort.index(cur_score)
                        if metric == Metric.DSC:
                            new_value *= 2

                        if len(gtv_list) > 1:
                            scores[epoch][stats][gtv][metric] = new_value
                        else:
                            scores[epoch][stats][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stats in ["avg", "median"]:
                for gtv in gtv_list:
                    for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                        if len(gtv_list) > 1:
                            evaluation[epoch] += scores[epoch][stats][gtv][metric]
                        else:
                            evaluation[epoch] += scores[epoch][stats][metric]

        return evaluation.key_with_max_value()

    def _inference_single_patient_record_labels(self, labels: Dict):
        outputs = Dict()
        for gtv in ["gtvt", "gtvn", "gtvs"]:
            outputs[gtv]["label"] = labels[gtv]
        return outputs

    def _inference_single_patient_record_outputs(
        self, outputs: Dict, preds: Dict, input_imgs: Tensor, idl_gtvn_clicks: Tensor
    ):
        outputs["gtvt"]["pred"] = preds[1]
        outputs["gtvn"]["pred"] = preds[2]
        outputs["gtvs"]["pred"] = np.maximum(preds[1], preds[2])
