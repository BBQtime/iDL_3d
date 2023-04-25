from custom import Global as g
import os
import torch
import math
import statistics
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from training import Training
from matplotlib import pyplot as plt
from baseline_dataset import BaselineDataSet
from loss_func import UnifiedFocalLoss
from custom import Dict
from custom import Json
from custom import List
from custom import Nii
from custom import Folder
from custom import GPU
from custom import Value
from custom import Explorer
from custom import Img


class BaselineTraining(Training):
    def _load_dataset(self, fold: int, debug_mode: bool = False):
        dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH)

        if len(dataset_split) - 1 != g.DATASET_K_FOLDS:
            dataset_split = super()._split_dataset()

        test_patients = List(dataset_split["test.set"])
        valid_patients = List(dataset_split["fold.{}".format(fold)])

        train_patients = List()
        for i in dataset_split:
            if i != "test.set" and i != "fold.{}".format(fold):
                train_patients += List(dataset_split[i])

        if debug_mode:
            train_patients = train_patients[:2]
            # 2 patients in valid and test sets, to debug median score calculation
            valid_patients = valid_patients[:2]
            test_patients = test_patients[:2]

        return train_patients, valid_patients, test_patients

    def __load_hyper(
        self,
        hyper: Dict,
        fold: int,
        cnn_path: str = None,  # make cnn_path == None or "" will load a new cnn
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # cross valid folds
        hyper["dataset.k.folds"] = g.DATASET_K_FOLDS

        # epochs
        if debug_mode:
            # at least 2 epochs to compare loss difference
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

        # load shared hyper parameters
        super()._load_hyper(hyper=hyper, cnn_path=cnn_path)

        # loss function
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
            training_type="baseline",
        ).to(g.DEVICE)

        # load patients
        (train_patients, valid_patients, test_patients,) = self._load_dataset(
            fold=fold,
            debug_mode=debug_mode,
        )

        hyper["dataset.len.train"] = train_patients.__len__()
        hyper["dataset.len.valid"] = valid_patients.__len__()
        hyper["dataset.len.test"] = test_patients.__len__()
        hyper["dataset.len"] = (
            train_patients.__len__()
            + valid_patients.__len__()
            + test_patients.__len__()
        )

        # create datasets
        # run this after shared hyper loaded, because hyper["augment"] is needed
        augment = Dict()
        augment["methods"] = hyper["augment.methods"]
        augment["pct"] = hyper["augment.pct"]
        augment["min"] = hyper["augment.min"]
        augment["max"] = hyper["augment.max"]

        train_set = BaselineDataSet(patients=train_patients, augment=augment)
        valid_set = BaselineDataSet(patients=valid_patients)
        test_set = BaselineDataSet(patients=test_patients)

        # dataloader
        hyper["train.loader"] = DataLoader(
            dataset=train_set,
            batch_size=hyper["batch.size.actual"],
            shuffle=True,  # only shuffle train loader
            num_workers=g.NUM_WORKERS,
        )
        hyper["valid.loader"] = DataLoader(
            dataset=valid_set,
            batch_size=hyper["batch.size.actual"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )
        hyper["test.loader"] = DataLoader(
            dataset=test_set,
            batch_size=hyper["batch.size.actual"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )

    def __get_simple_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for cur_key in hyper:
            if (
                cur_key == "train.loader"
                or cur_key == "valid.loader"
                or cur_key == "test.loader"
            ):
                pass
            else:
                simple_hyper[cur_key] = hyper[cur_key]
        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._save_hyper(simple_hyper, json_path)

    def plot_lr_fig(self, baseline_id: str):
        lr_json_path = os.path.join(
            g.TRAIN_RESULTS_DIR, baseline_id, "baseline", "lr.json"
        )
        self._plot_lr_fig(lr_json_path)

    # protected function, idl_gtvn_training will inherit it
    def _plot_lr_fig(self, lr_json_path: str):
        plt.figure().clear()

        lr_dict = Json.load(lr_json_path)
        lr_list = List()
        for i in lr_dict:
            lr_list.append(lr_dict[i])

        plt.plot(range(1, len(lr_list) + 1), lr_list, label="lr")
        plt.legend()
        plt.savefig(lr_json_path[:-4] + "png")

    def plot_loss_fig(self, baseline_id: str):
        loss_json_path = os.path.join(
            g.TRAIN_RESULTS_DIR, baseline_id, "baseline", "loss.json"
        )
        self._plot_loss_fig(loss_json_path)

    # protected function, idl_gtvn_training will inherit it
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

    def __training(self, hyper: Dict, fold_dir: str):
        best_loss_dict = Dict()
        loss_json_path = os.path.join(fold_dir, "train_info", "loss.json")
        lr_json_path = os.path.join(fold_dir, "train_info", "lr.json")
        patience = 0

        for epoch in range(1, hyper["epochs"] + 1):
            print("")
            print("epoch: {}".format(epoch))
            print("training:")
            hyper["cnn"].train()
            train_loss = 0
            num_batches = 0
            for multimodal_imgs, labels in tqdm(hyper["train.loader"]):
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                multimodal_imgs = multimodal_imgs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                preds = hyper["cnn"](multimodal_imgs)
                cur_loss = hyper["loss.func"](preds, labels, weight_map=None)
                cur_loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                train_loss += cur_loss.item()
                num_batches += 1
            train_loss /= num_batches

            # validation
            print("validation:")
            hyper["cnn"].eval()
            with torch.no_grad():
                valid_loss = 0
                num_batches = 0
                for multimodal_imgs, labels in tqdm(hyper["valid.loader"]):
                    multimodal_imgs = multimodal_imgs.to(g.DEVICE)
                    labels = labels.to(g.DEVICE)
                    preds = hyper["cnn"](multimodal_imgs)
                    cur_loss = hyper["loss.func"](preds, labels, weight_map=None)
                    valid_loss += cur_loss.item()
                    num_batches += 1
            valid_loss /= num_batches
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
                Folder.create(os.path.join(epoch_dir, "baseline"))
                cnn_save_path = os.path.join(
                    epoch_dir,
                    "baseline",
                    "epoch={:03d}.pt".format(epoch),
                )
                self._save_cnn(hyper, cnn_save_path)
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
                    Folder.create(os.path.join(epoch_dir, "baseline"))
                    cnn_save_path = os.path.join(
                        epoch_dir,
                        "baseline",
                        "epoch={:03d}.pt".format(epoch),
                    )
                    self._save_cnn(hyper, cnn_save_path)
                    patience = 0
                else:
                    patience += 1
                    if patience >= hyper["early.stop.epochs"]:
                        break

    def new_training(
        self,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        for hyper in self._load_hyper_list_from_json(g.HYPER_JSON_PATH_BASELINE):

            baseline_id = "baseline_" + self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.HYPER_JSON_PATH_BASELINE,
                hyper=hyper,
                debug_mode=debug_mode,
            )
            print("")
            print(baseline_id)

            baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)
            Folder.create(baseline_dir)

            for fold in range(1, g.DATASET_K_FOLDS + 1):
                fold_dir = os.path.join(baseline_dir, "fold={:02d}".format(fold))
                Folder.create(fold_dir)

                self.__load_hyper(
                    hyper=hyper,
                    fold=fold,
                    cnn_path="",  # cnn_path="" will create a new cnn
                    debug_mode=debug_mode,
                )
                if fold == 1:
                    print("")
                    self._print_hyper(hyper)

                print("")
                print("cross validation fold: {}".format(fold))

                train_info_dir = os.path.join(fold_dir, "train_info")
                Folder.create(train_info_dir)
                # save an empty loss.json
                Json.save(Dict(), os.path.join(train_info_dir, "loss.json"))
                # save an empty lr.json
                Json.save(Dict(), os.path.join(train_info_dir, "lr.json"))

                # save hyper before training
                hyper_json_path = os.path.join(train_info_dir, "hyper.json")
                self._save_hyper(hyper, hyper_json_path)

                # start training
                hyper["time.spent"] = datetime.now()
                self.__training(hyper, fold_dir)
                hyper["time.spent"] = datetime.now() - hyper["time.spent"]
                hyper["time.spent"] = str(hyper["time.spent"]).split(".", 2)[0]

                # save hyper after training
                self._save_hyper(hyper, hyper_json_path)

                # clear time spent before next training
                hyper.pop("time.spent")

                # break if no cross validation
                if hyper["dataset.cross.valid"] is False:
                    break
                # only train 2 folds in debug mode
                if debug_mode and fold == 2:
                    break

            # inference (valid first, then test, train comes last)
            for dataset in ["valid", "test", "train"]:
                self.inference(
                    baseline_id=baseline_id,
                    dataset=dataset,
                    debug_mode=debug_mode,
                )

    def __patient_inference(self, patient: str, hyper: Dict) -> Dict:
        # result structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
        results = Dict()
        origin_labels = Dict()
        dataset = BaselineDataSet(patients=[patient])

        # load origin_labels
        for i in ["s", "t", "n"]:
            origin_labels["gtv{}".format(i)] = Nii.load(
                os.path.join(g.DATASET_DIR, "HNCDL_{}_GTV{}.nii".format(patient, i)),
                binary=True,
            )

        # get pred
        hyper["cnn"].eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            multimodal_imgs, labels = dataset.get_item(patient=patient)
            multimodal_imgs = torch.unsqueeze(multimodal_imgs.to(g.DEVICE), dim=0)
            labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
            preds = hyper["cnn"].forward(multimodal_imgs)
            # squeeze "batch" channel
            preds = torch.squeeze(preds, dim=0).cpu().numpy()

        results["gtvt"]["pred"] = preds[1]
        results["gtvn"]["pred"] = preds[2]
        results["gtvs"]["pred"] = np.maximum(preds[1], preds[2])

        # pad and crop to original size
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            results[gtv]["pred"] = Img.central_pad(
                results[gtv]["pred"], origin_labels[gtv].shape
            )
            results[gtv]["pred"] = Img.central_crop(
                results[gtv]["pred"], origin_labels[gtv].shape
            )
            # calculate segment scores
            for metric in g.METRICS:
                results[gtv][metric] = self._metrics[metric](
                    results[gtv]["pred"], origin_labels[gtv]
                )
        return results

    def inference(
        self,
        baseline_id: str,
        dataset: str = "test",  # train/valid/test
        debug_mode: bool = False,
    ):
        print("")
        print("inference on {} set: {}".format(dataset, baseline_id))

        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)

        if dataset != "train" and dataset != "valid":
            dataset = "test"

        # if on valid set, delete non-optimal epoch results
        if dataset == "valid":
            # record top median scores through each fold and epoch
            top_median_scores_set = Dict()

        # loop through fold folders
        for fold_dir in Explorer.get_sub_folders(
            baseline_dir, key_word="fold=", return_full_path=True
        ):
            fold = int(Path(fold_dir).name[len("fold=") :])
            print("")
            print("current fold: ", fold)
            fold_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, fold_dir)

            # loop through epoch folders
            for epoch_dir in Explorer.get_sub_folders(
                fold_dir, key_word="epoch=", return_full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("current epoch: ", epoch)
                epoch_dir = os.path.join(fold_dir, epoch_dir)

                # initialize scores dict (only for test and valid set)
                if dataset == "test" or dataset == "valid":
                    epoch_scores = Dict()
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric in g.METRICS:
                            epoch_scores["median"][gtv][metric] = List()

                # load cnn
                cnn_path = Explorer.get_sub_files(
                    os.path.join(epoch_dir, "baseline"),
                    key_word=".pt",
                    return_full_path=True,
                )[0]
                hyper = Dict()  # create an empty hyper dict to save cnn
                self._load_cnn(hyper=hyper, cnn_path=cnn_path)

                # load patients
                train_patients, valid_patients, test_patients = self._load_dataset(
                    fold=fold, debug_mode=debug_mode
                )
                if dataset == "test":
                    patients = test_patients
                elif dataset == "valid":
                    patients = valid_patients
                elif dataset == "train":
                    patients = train_patients

                for patient in tqdm(patients):
                    # create folder to save cur patient preds and scores
                    patient_dir = os.path.join(
                        epoch_dir,
                        "baseline",
                        "patients",
                        "patient={}".format(patient),
                    )
                    Folder.create(patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self.__patient_inference(
                        patient=patient, hyper=hyper
                    )

                    # save gtvt and gtvn preds of each patient
                    for gtv in ["gtvt", "gtvn"]:
                        Nii.save(
                            img=patient_results[gtv]["pred"],
                            path=os.path.join(patient_dir, "pred_{}.nii".format(gtv)),
                            spacing=g.NII_SPACING,
                        )

                    # record score of current patient
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric in g.METRICS:
                            # copy cur patient score (test set only)
                            if dataset == "test":
                                epoch_scores["patient={}".format(patient)][gtv][
                                    metric
                                ] = patient_results[gtv][metric]
                            # add scores of current patient into median(list)
                            if dataset == "test" or dataset == "valid":
                                epoch_scores["median"][gtv][metric].append(
                                    patient_results[gtv][metric]
                                )

                    # save current patient scores to the patient dir
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        # remove pred as it can't be save in json
                        # make sure it has already been saved
                        patient_results[gtv].pop("pred")
                    Json.save(
                        data=patient_results,
                        path=os.path.join(patient_dir, "inference.json"),
                    )

                # all patients under current epoch have been traversed
                # dont need to do anything on training set
                if dataset == "train":
                    continue

                # calculate median score (test and valid set only)
                if dataset == "test" or dataset == "valid":
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric in g.METRICS:
                            median = epoch_scores["median"][gtv][metric]
                            epoch_scores["median"][gtv][metric] = statistics.median(
                                median
                            )

                # save all patients scores in "baseline" dir
                if dataset == "test":
                    Json.save(
                        data=epoch_scores,
                        path=os.path.join(
                            epoch_dir, "baseline", "inference_{}.json".format(dataset)
                        ),
                    )
                    continue

                # valid set, delete non-optimal folds and epochs
                if dataset == "valid":
                    epoch_median_scores = epoch_scores["median"]
                    if math.isnan(epoch_median_scores["gtvs"]["msd"]) or math.isnan(
                        epoch_median_scores["gtvs"]["hd95"]
                    ):
                        Folder.delete(epoch_dir)
                        continue

                    if top_median_scores_set == {}:
                        keep_epoch_results = True
                    else:
                        # assign keep_epoch_results to None,
                        # it will become True/Flase sooner or later
                        keep_epoch_results = None

                        # loop through top_median_scores_set
                        for fold_key in top_median_scores_set.keys():
                            for epoch_key in top_median_scores_set[fold_key].keys():

                                # skip cur_epoch_scores,
                                # if it is worse than any of the top_median_scores_set
                                if (
                                    epoch_median_scores["gtvs"]["dsc"]
                                    < top_median_scores_set[fold_key][epoch_key]["dsc"]
                                    and epoch_median_scores["gtvs"]["msd"]
                                    > top_median_scores_set[fold_key][epoch_key]["msd"]
                                    and epoch_median_scores["gtvs"]["hd95"]
                                    > top_median_scores_set[fold_key][epoch_key]["hd95"]
                                ):
                                    keep_epoch_results = False
                                    break

                                # save cur_epoch_scores,
                                # if it is comparable for one of the top_median_scores_set
                                # (comparable means: at least one of dsc/msd/hd95 is better)
                                if (
                                    epoch_median_scores["gtvs"]["dsc"]
                                    > top_median_scores_set[fold_key][epoch_key]["dsc"]
                                    or epoch_median_scores["gtvs"]["msd"]
                                    < top_median_scores_set[fold_key][epoch_key]["msd"]
                                    or epoch_median_scores["gtvs"]["hd95"]
                                    < top_median_scores_set[fold_key][epoch_key]["hd95"]
                                ):
                                    keep_epoch_results = True

                                # delete any score in top_median_scores_set,
                                # if it is worse than cur_epoch_scores
                                if (
                                    top_median_scores_set[fold_key][epoch_key]["dsc"]
                                    < epoch_median_scores["gtvs"]["dsc"]
                                    and top_median_scores_set[fold_key][epoch_key][
                                        "msd"
                                    ]
                                    > epoch_median_scores["gtvs"]["msd"]
                                    and top_median_scores_set[fold_key][epoch_key][
                                        "hd95"
                                    ]
                                    > epoch_median_scores["gtvs"]["hd95"]
                                ):
                                    Folder.delete(
                                        os.path.join(
                                            baseline_dir,
                                            "fold={:02d}".format(fold_key),
                                            "epoch={:03d}".format(epoch_key),
                                        )
                                    )
                            if keep_epoch_results is False:
                                break

                    # save cur epoch median scores
                    if keep_epoch_results:
                        top_median_scores_set[fold][epoch] = epoch_median_scores["gtvs"]
                        Json.save(
                            data=epoch_scores,
                            path=os.path.join(
                                epoch_dir, "baseline", "inference_valid.json"
                            ),
                        )
                    else:
                        Folder.delete(epoch_dir)
