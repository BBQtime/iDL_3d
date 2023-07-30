import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from custom import GPU, Debug, Dict, Directory, Folder
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from dataset_baseline import DataSetBaseline
from loss_func import UnifiedFocalLoss
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_core import TrainingCore


class TrainingBaseline(TrainingCore):
    def _load_hyper(
        self,
        hyper: Dict,
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
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

        # load patients
        patients = self._load_patients(fold=hyper["fold"], debug_mode=debug_mode)
        for i in ["train", "valid", "test.inter"]:
            hyper["{}.patients".format(i)] = patients[i]

        self._load_slice_thick(hyper)

        self._load_loss_func(hyper)

        # load datasets before load dataloaders
        self._load_datasets(hyper)

        self._load_data_loaders(hyper)

        # load cnn before optimizer
        self._load_new_cnn(hyper=hyper)

        self._load_optim_and_scheduler(hyper=hyper)

    def _load_new_cnn(self, hyper: Dict, in_chan: int = 4, out_chan: int = 3):
        super()._load_new_cnn(hyper=hyper, in_chan=in_chan, out_chan=out_chan)

    def _load_loss_func(self, hyper: Dict):
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

    def _load_datasets(self, hyper: Dict):
        # load train/valid/test datasets
        for i in ["train", "valid", "test.inter"]:
            # only use data augmentation on training set
            if i == "train":
                augment = Dict()
                augment["methods"] = hyper["augment.methods"]
                augment["pct"] = hyper["augment.pct"]
                augment["min"] = hyper["augment.min"]
                augment["max"] = hyper["augment.max"]
            else:
                augment = None
            hyper["{}.set".format(i)] = DataSetBaseline(
                patients=hyper["{}.patients".format(i)],
                slice_thick=hyper["slice.thick"],
                augment=augment,
            )

    # baseline/idl.gtvn/idl.gtvs share this function
    def _load_data_loaders(self, hyper: Dict):
        for i in ["train", "valid", "test.inter"]:
            # only shuffle train loader
            if i == "train":
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
        for i in ["train", "valid", "test.inter"]:
            ignore_list.append("{}.patients".format(i))
            ignore_list.append("{}.set".format(i))
            ignore_list.append("{}.loader".format(i))

        # here in this for loop, use "hyper" instead of "simple_hyper"
        # otherwise will cause error: dictionary changed size during iteration
        for key_name in hyper:
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
            train_loss.append(loss_dict[i]["train"])
            valid_loss.append(loss_dict[i]["valid"])

        # draw figure
        plt.figure().clear()
        plt.ylim(min(train_loss) - 0.05, max(train_loss) + 0.05)
        plt.plot(range(1, len(loss_dict) + 1), train_loss, label="train")
        plt.plot(range(1, len(loss_dict) + 1), valid_loss, label="valid")
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
            epoch_loss["train"] = train_loss
            epoch_loss["valid"] = valid_loss
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
        self,
        hyper: Dict,
        train_dir: str,
        debug_mode: bool = False,
    ):
        Folder.create(train_dir)

        # cross validation
        hyper["fold"] = int(hyper["fold"])
        hyper["fold"] = Value.limit_range(hyper["fold"], (0, g.DATASET_FOLDS))
        # fold=0 will activate cross validation
        if hyper["fold"] == 0:
            fold_list = List(range(1, g.DATASET_FOLDS + 1))
        else:
            fold_list = [hyper["fold"]]

        # backup origin hyper for resetting hyper on next fold
        # (after "fold" removed from hyper Dict)
        hyper.pop("fold")
        origin_hyper = hyper.copy()

        # loop through each fold
        for hyper["fold"] in fold_list:
            fold_dir = os.path.join(train_dir, "fold={}".format(hyper["fold"]))
            Folder.create(fold_dir)

            # load and print hyperparams
            self._load_hyper(hyper=hyper, debug_mode=debug_mode)
            print("")
            self._print_hyper(hyper)

            print("")
            print("fold: {}".format(hyper["fold"]))

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
        for hyper in self._load_hyper_sets_from_json(g.HYPER_JSON_PATH_BASELINE):
            baseline_id = "baseline_" + self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.HYPER_JSON_PATH_BASELINE,
                hyper=hyper,
                debug_mode=debug_mode,
            )
            print("")
            print(baseline_id)
            baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, "baseline")

            self._training_all_folds(
                hyper=hyper, train_dir=baseline_dir, debug_mode=debug_mode
            )

            # inference
            self.inference(baseline_id=baseline_id, debug_mode=debug_mode)

            # after inference
            self.remove_non_optimal_epochs(baseline_id)

            # after non optimal epochs removed
            self.calculate_cross_valid_scores(
                baseline_id=baseline_id, debug_mode=debug_mode
            )

    def inference(self, baseline_id: str, debug_mode: bool = False):
        print("")
        print("inference: {}".format(baseline_id))

        # find idl gtvt folder
        baseline_dir = self._find_train_dir(baseline_id)
        if baseline_dir is None:
            Debug.error_exit("baseline_id not found")

        fold_dirs = Directory.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        )

        # load slice thickness and segmentation metrics
        slice_thick = Json.load(os.path.join(fold_dirs[0], "hyper.json"))["slice.thick"]
        segment_metrics = self._load_segment_metrics(slice_thick)

        # loop through fold dirs
        for fold_dir in fold_dirs:
            fold = int(Path(fold_dir).name[len("fold=") :])
            print("")
            print("fold: ", fold)

            # loop through epoch dirs
            for epoch_dir in Directory.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("epoch: ", epoch)

                # load cnn
                cnn_path = os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch))
                cnn = self._load_exist_cnn(cnn_path)

                # initialize scores dict
                epoch_scores = Dict()
                for stats in ["median", "avg"]:
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric in g.METRICS:
                            epoch_scores[stats][gtv][metric] = List()

                # load patients
                all_patients = self._load_patients(debug_mode=debug_mode)
                inter_test_patients = all_patients["test.inter"]
                all_patients = all_patients.to_list()

                for patient in tqdm(all_patients):
                    # create folder to save cur patient preds and scores
                    patient_dir = os.path.join(
                        epoch_dir,
                        "patients",
                        "patient={}".format(patient),
                    )
                    Folder.create(patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self.__single_patient_inference(
                        patient=patient, cnn=cnn, slice_thick=slice_thick
                    )

                    # save preds of current patient
                    for gtv in ["gtvt", "gtvn"]:
                        Nii.save(
                            img=patient_results[gtv]["pred"],
                            save_path=os.path.join(
                                patient_dir, "{}_pred.nii".format(gtv)
                            ),
                            spacing=g.NII_SPACING[slice_thick],
                        )

                    # record score of current patient (test set only)
                    if patient in inter_test_patients:
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                score = segment_metrics[metric](
                                    patient_results[gtv]["pred"],
                                    patient_results[gtv]["label"],
                                )
                                # save cur patient score
                                epoch_scores["patient={}".format(patient)][gtv][
                                    metric
                                ] = score
                                # add scores of current patient into avg and median
                                for stats in ["median", "avg"]:
                                    epoch_scores[stats][gtv][metric].append(score)

                # all patients under current epoch have been traversed
                # calculate median and avg score of current epoch
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in g.METRICS:
                        epoch_scores["median"][gtv][metric] = Value.median(
                            epoch_scores["median"][gtv][metric]
                        )
                        epoch_scores["avg"][gtv][metric] = Value.avg(
                            epoch_scores["avg"][gtv][metric]
                        )
                # save all patients scores in "inference_test_inter.json"
                Json.save(
                    data=epoch_scores,
                    path=os.path.join(epoch_dir, "inference_test_inter.json"),
                )
                continue  # next epoch

    def calculate_cross_valid_scores(self, baseline_id: str, debug_mode: bool = False):
        print("")
        print("calculate cross valid scores: {}".format(baseline_id))

        baseline_dir = self._find_train_dir(baseline_id)

        Folder.create(os.path.join(baseline_dir, "patients"))

        fold_dirs = Directory.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        )

        # load slice thickness and segmentation metrics
        slice_thick = Json.load(os.path.join(fold_dirs[0], "hyper.json"))["slice.thick"]
        segment_metrics = self._load_segment_metrics(slice_thick)

        # initialize scores dict
        scores = Dict()
        for stats in ["median", "avg"]:
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in g.METRICS:
                    scores[stats][gtv][metric] = List()

        # load patients
        all_patients = self._load_patients(debug_mode=debug_mode)
        inter_test_patients = all_patients["test.inter"]
        all_patients = all_patients.to_list()

        for patient in tqdm(all_patients):
            # initialize preds
            preds = Dict()
            for gtv in ["s", "t", "n"]:
                preds[gtv] = None

            for fold_dir in fold_dirs:
                # find epoch dir
                epoch_dirs = Directory.get_sub_folders(
                    fold_dir, key_word="epoch=", full_path=True
                )
                if len(epoch_dirs) > 1:
                    self.remove_non_optimal_epochs(baseline_id)
                    epoch_dir = Directory.get_sub_folders(
                        fold_dir, key_word="epoch=", full_path=True
                    )[0]
                else:
                    epoch_dir = epoch_dirs[0]

                # load preds
                patient_dir = os.path.join(
                    epoch_dir, "patients", "patient={}".format(patient)
                )
                for gtv in ["t", "n"]:
                    img = Nii.load(
                        os.path.join(patient_dir, "gtv{}_pred.nii".format(gtv))
                    )
                    if preds[gtv] is None:
                        preds[gtv] = img
                    else:
                        preds[gtv] += img

            preds["s"] = preds["t"] + preds["n"]
            for gtv in ["s", "t", "n"]:
                preds[gtv] /= len(fold_dirs)

            # save cross_valid preds
            patient_dir = os.path.join(
                baseline_dir, "patients", "patient={}".format(patient)
            )
            Folder.create(patient_dir)
            for gtv in ["t", "n"]:
                Nii.save(
                    img=preds[gtv],
                    save_path=os.path.join(patient_dir, "gtv{}_pred.nii".format(gtv)),
                    spacing=g.NII_SPACING[slice_thick],
                )

            # load labels and calculate metrics (on internal test set only)
            if patient in inter_test_patients:
                labels = Dict()
                for gtv in ["s", "t", "n"]:
                    labels[gtv] = Nii.load(
                        os.path.join(
                            g.DATASET_DIR[slice_thick],
                            "HNCDL_{}_GTV{}.nii".format(patient, gtv),
                        ),
                        binary=True,
                    )
                    for metric in g.METRICS:
                        score = segment_metrics[metric](preds[gtv], labels[gtv])
                        scores["patient={}".format(patient)]["gtv{}".format(gtv)][
                            metric
                        ] = score

                        for stats in ["median", "avg"]:
                            scores[stats]["gtv{}".format(gtv)][metric].append(score)

        # calculate avg and median score
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            for metric in g.METRICS:
                scores["median"][gtv][metric] = Value.median(
                    scores["median"][gtv][metric]
                )
                scores["avg"][gtv][metric] = Value.avg(scores["avg"][gtv][metric])

        # save cross valid scores
        Json.save(
            data=scores,
            path=os.path.join(baseline_dir, "inference_test_inter.json"),
        )

    def remove_non_optimal_epochs(self, baseline_id: str):
        baseline_dir = self._find_train_dir(baseline_id)
        print("")
        print("remove non optimal epochs: {}".format(baseline_id))

        for fold_dir in Directory.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        ):
            fold_scores = Dict()

            for epoch_dir in Directory.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name

                # load scores of current epoch
                # if baseline
                epoch_scores = Json.load(
                    os.path.join(epoch_dir, "inference_test_inter.json")
                )
                for stats in ["median", "avg"]:
                    fold_scores[epoch][stats] = epoch_scores[stats]

            best_epoch = self.__find_best_result(fold_scores)

            # delete non-optimal epochs
            for epoch_dir in Directory.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                if epoch != best_epoch:
                    Folder.delete(epoch_dir)
                    print("delete: {} {}".format(Path(fold_dir).name, epoch))

    # a sub function of _remove_non_optimal_epochs()
    def __find_best_result(self, scores: Dict):
        for stats in ["median", "avg"]:
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in g.METRICS:
                    # create a tmp list to sort
                    list_to_sort = List()
                    # add elements into the list
                    for epoch in scores.keys():
                        list_to_sort.append(scores[epoch][stats][gtv][metric])
                    # sort the list
                    if metric == "dsc":
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)
                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        new_value = list_to_sort.index(
                            scores[epoch][stats][gtv][metric]
                        )
                        # if metric == "dsc":
                        #     new_value *= 2
                        scores[epoch][stats][gtv][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stats in ["avg", "median"]:
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in g.METRICS:
                        evaluation[epoch] += scores[epoch][stats][gtv][metric]

        return evaluation.key_with_max_value()

    def __single_patient_inference(self, patient: str, cnn, slice_thick: str) -> Dict:
        # result structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
        result = Dict()

        dataset = DataSetBaseline(patients=[patient], slice_thick=slice_thick)

        # load labels
        for gtv in ["s", "t", "n"]:
            result["gtv{}".format(gtv)]["label"] = Nii.load(
                os.path.join(
                    g.DATASET_DIR[slice_thick],
                    "HNCDL_{}_GTV{}.nii".format(patient, gtv),
                ),
                binary=True,
            )

        input_imgs, labels = dataset.get_item(patient=patient)

        # add "batch" (c/d/h/w -> b/c/d/h/w)
        input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)
        labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)

        cnn.eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            preds = cnn.forward(input_imgs)
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        preds = torch.squeeze(preds, dim=0).cpu().numpy()

        result["gtvt"]["pred"] = preds[1]
        result["gtvn"]["pred"] = preds[2]
        result["gtvs"]["pred"] = np.maximum(preds[1], preds[2])

        # pad and crop preds to original size
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            result[gtv]["pred"] = Img.central_resize(
                result[gtv]["pred"], result[gtv]["label"].shape
            )

        return result
