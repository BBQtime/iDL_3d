from custom import Global as g
import os
import torch
import math
import statistics
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from training import Training
from matplotlib import pyplot as plt
from baseline_dataset import BaselineDataSet
from idl_gtvn_dataset import IDLGTVnDataSet
from loss_func import UnifiedFocalLoss
from custom import Dict
from custom import Json
from custom import List
from custom import Nii
from custom import Folder
from custom import GPU
from custom import Value
from custom import Explorer


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

    def _load_hyper(
        self,
        hyper: Dict,
        fold: int,
        baseline_epoch_dir: str = None,  # this is only for idl.gtvn
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # cross valid folds
        hyper["cross.valid.fold"] = "{}/{}".format(fold, g.DATASET_K_FOLDS)

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
        super()._load_hyper(hyper=hyper, cnn_path=None)

        # loss function
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
            train_type=hyper["train.type"],
        ).to(g.DEVICE)

        # load patients
        (train_patients, valid_patients, test_patients,) = self._load_dataset(
            fold=fold,
            debug_mode=debug_mode,
        )

        # create datasets
        # run this after shared hyper loaded, because hyper["augment"] is needed
        augment = Dict()
        augment["methods"] = hyper["augment.methods"]
        augment["pct"] = hyper["augment.pct"]
        augment["min"] = hyper["augment.min"]
        augment["max"] = hyper["augment.max"]

        if hyper["train.type"] == "baseline":
            train_set = BaselineDataSet(patients=train_patients, augment=augment)
            valid_set = BaselineDataSet(patients=valid_patients)
            test_set = BaselineDataSet(patients=test_patients)
        else:
            train_set = IDLGTVnDataSet(
                patients=train_patients,
                baseline_epoch_dir=baseline_epoch_dir,
                augment=augment,
                random_click=False,
                # random_click=True,
            )
            valid_set = IDLGTVnDataSet(
                patients=valid_patients,
                baseline_epoch_dir=baseline_epoch_dir,
                random_click=False,
            )
            test_set = IDLGTVnDataSet(
                patients=test_patients,
                baseline_epoch_dir=baseline_epoch_dir,
                random_click=False,
            )

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

    def __simplify_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for key_name in hyper:
            # dont need to save or print dataloaders
            if (
                key_name == "train.loader"
                or key_name == "valid.loader"
                or key_name == "test.loader"
            ):
                pass
            # others
            else:
                simple_hyper[key_name] = hyper[key_name]
        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self.__simplify_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self.__simplify_hyper(hyper)
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

    def _training(self, hyper: Dict, fold_dir: str):
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
            for input_imgs, labels in tqdm(hyper["train.loader"]):
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                input_imgs = input_imgs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                preds = hyper["cnn"](input_imgs)
                loss = hyper["loss.func"](preds, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                train_loss += loss.item()
                num_batches += 1
            train_loss /= num_batches

            # validation
            print("validation:")
            hyper["cnn"].eval()
            with torch.no_grad():
                valid_loss = 0
                num_batches = 0
                for input_imgs, labels in tqdm(hyper["valid.loader"]):
                    input_imgs = input_imgs.to(g.DEVICE)
                    labels = labels.to(g.DEVICE)
                    preds = hyper["cnn"](input_imgs)
                    loss = hyper["loss.func"](preds, labels)
                    valid_loss += loss.item()
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
                if hyper["train.type"] == "baseline":
                    cnn_save_dir = os.path.join(epoch_dir, "baseline")
                else:
                    cnn_save_dir = epoch_dir
                Folder.create(cnn_save_dir)
                self._save_cnn(
                    hyper,
                    os.path.join(cnn_save_dir, "epoch={:03d}.pt".format(epoch)),
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
                    if hyper["train.type"] == "baseline":
                        cnn_save_dir = os.path.join(epoch_dir, "baseline")
                    else:
                        cnn_save_dir = epoch_dir
                    Folder.create(cnn_save_dir)
                    self._save_cnn(
                        hyper,
                        os.path.join(cnn_save_dir, "epoch={:03d}.pt".format(epoch)),
                    )
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
        self._new_training(
            train_type="baseline",
            train_remark=train_remark,
            debug_mode=debug_mode,
        )

    def _new_training(
        self,
        train_type: str,
        baseline_id: str = None,
        baseline_fold: int = None,
        baseline_epoch: int = None,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        if train_type == "baseline":
            hyper_json_path = g.HYPER_JSON_PATH_BASELINE
        else:
            train_type = "idl_gtvn"
            hyper_json_path = g.HYPER_JSON_PATH_IDL_GTVN

        for hyper in self._load_hyper_sets_from_json(hyper_json_path):

            # add training type into hyper
            hyper["train.type"] = train_type

            train_id = hyper["train.type"] + "_"
            train_id += self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=hyper_json_path,
                hyper=hyper,
                debug_mode=debug_mode,
            )
            print("")
            print(train_id)

            # find baseline fold dir
            if hyper["train.type"] == "idl_gtvn":
                if baseline_fold is None or baseline_fold <= 0:
                    key_word = "fold="
                else:
                    key_word = "fold={}".format(baseline_fold)
                baseline_fold_dir = Explorer.get_sub_folders(
                    os.path.join(g.TRAIN_RESULTS_DIR, baseline_id),
                    key_word=key_word,
                    return_full_path=True,
                )[0]
                # find epoch folder
                if baseline_epoch is None or baseline_epoch <= 0:
                    key_word = "epoch="
                else:
                    key_word = "epoch={:03d}".format(baseline_epoch)
                baseline_epoch_dir = Explorer.get_sub_folders(
                    baseline_fold_dir, key_word=key_word, return_full_path=True
                )[0]
            else:
                baseline_epoch_dir = None

            # create train result dir
            if hyper["train.type"] == "baseline":
                train_result_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id)
            else:
                train_result_dir = os.path.join(
                    baseline_epoch_dir, "idl_gtvn", train_id
                )
            Folder.create(train_result_dir)

            # cross validation
            hyper["cross.valid.fold"] = int(hyper["cross.valid.fold"])
            hyper["cross.valid.fold"] = Value.limit_range(
                hyper["cross.valid.fold"], (0, g.DATASET_K_FOLDS)
            )
            if hyper["cross.valid.fold"] == 0:
                fold_list = List(range(1, g.DATASET_K_FOLDS + 1))
            else:
                fold_list = [hyper["cross.valid.fold"]]

            # loop through each fold
            for fold in fold_list:
                fold_dir = os.path.join(train_result_dir, "fold={}".format(fold))
                Folder.create(fold_dir)

                # load and print hyperparams
                self._load_hyper(
                    hyper=hyper,
                    fold=fold,
                    baseline_epoch_dir=baseline_epoch_dir,
                    debug_mode=debug_mode,
                )
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
                hyper_save_path = os.path.join(train_info_dir, "hyper.json")
                self._save_hyper(hyper, hyper_save_path)

                # start training
                hyper["time.spent"] = datetime.now()
                self._training(hyper, fold_dir)
                hyper["time.spent"] = datetime.now() - hyper["time.spent"]
                hyper["time.spent"] = str(hyper["time.spent"]).split(".", 2)[0]

                # save hyper after training
                self._save_hyper(hyper, hyper_save_path)

                # clear time spent before next training
                hyper.pop("time.spent")

                # only train 2 folds in debug mode
                if debug_mode and fold_list.index(fold) == 1:
                    break

            # inference (valid set first)
            for dataset in ["valid", "test"]:
                self._inference(
                    train_id=train_id,
                    dataset=dataset,
                    debug_mode=debug_mode,
                )

    def inference(
        self,
        baseline_id: str,
        dataset: str = "test",  # train/valid/test
        debug_mode: bool = False,
    ):
        if dataset != "train" and dataset != "valid":
            dataset = "test"
        self._inference(
            train_id=baseline_id,
            dataset=dataset,
            debug_mode=debug_mode,
        )

    def _inference(
        self,
        train_id: str,
        dataset: str = "test",  # train/valid/test
        debug_mode: bool = False,
    ):

        # confirm inference_type
        inference_type = None
        for i in ["baseline", "idl_gtvn"]:
            if i in train_id:
                inference_type = i
        if inference_type is None:
            print("""cant find key word "baseline" or "idl_gtvn" in train_id""")
            return

        print("")
        print("inference on {} set: {}".format(dataset, train_id))

        # find idl gtvt folder
        if inference_type == "baseline":
            train_result_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id)
            if not os.path.exists(train_result_dir):
                train_result_dir = None
        else:
            train_result_dir = self._find_result_dir(train_id)
        if train_result_dir is None:
            print("train_id not found")
            return

        # this is only for idl_gtvn to load baseline gtvn preds
        if inference_type == "idl_gtvn":
            idl_gtvn_baseline_epoch_dir = str(Path(train_result_dir).parent.parent)
        else:
            idl_gtvn_baseline_epoch_dir = None

        # loop through fold dirs
        for fold_dir in Explorer.get_sub_folders(
            train_result_dir, key_word="fold=", return_full_path=True
        ):
            fold = int(Path(fold_dir).name[len("fold=") :])
            print("")
            print("fold: ", fold)

            # loop through epoch dirs
            for epoch_dir in Explorer.get_sub_folders(
                fold_dir, key_word="epoch=", return_full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("epoch: ", epoch)

                # load cnn
                if inference_type == "baseline":
                    cnn_dir = os.path.join(epoch_dir, "baseline")
                elif inference_type == "idl_gtvn":
                    cnn_dir = epoch_dir
                cnn_path = Explorer.get_sub_files(
                    cnn_dir, key_word=".pt", return_full_path=True
                )[0]
                hyper = Dict()  # create an empty hyper dict to save cnn
                self._load_cnn(hyper=hyper, cnn_path=cnn_path)

                # load dataset patients
                train_patients, valid_patients, test_patients = self._load_dataset(
                    fold=fold, debug_mode=debug_mode
                )
                if dataset == "test":
                    patients = test_patients
                elif dataset == "valid":
                    patients = valid_patients
                elif dataset == "train":
                    patients = train_patients

                # initialize scores dict (only on test and valid set)
                if dataset == "test" or dataset == "valid":
                    epoch_scores = Dict()
                    if inference_type == "baseline":
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                epoch_scores["median"][gtv][metric] = List()
                    else:
                        if dataset == "valid":
                            # no need to record baseline score for valid set
                            for metric in g.METRICS:
                                epoch_scores["median"][metric] = List()
                        elif dataset == "test":
                            # record baseline score in ["round=00"] for test set
                            # so here, initialize ["round=01"] as a list
                            for metric in g.METRICS:
                                epoch_scores["median"][metric]["round=01"] = List()
                            # copy baseline scores of each patient
                            baseline_scores = Json.load(
                                os.path.join(
                                    idl_gtvn_baseline_epoch_dir,
                                    "baseline",
                                    "inference_test.json",
                                )
                            )
                            for patient in patients:
                                for metric in g.METRICS:
                                    epoch_scores["patient={}".format(patient)][metric][
                                        "round=00"
                                    ] = baseline_scores["patient={}".format(patient)][
                                        "gtvn"
                                    ][
                                        metric
                                    ]
                            # also copy median score of each patient
                            for metric in g.METRICS:
                                epoch_scores["median"][metric][
                                    "round=00"
                                ] = baseline_scores["median"]["gtvn"][metric]

                for patient in tqdm(patients):
                    # create folder to save cur patient preds and scores
                    if inference_type == "baseline":
                        patient_dir = os.path.join(
                            epoch_dir,
                            "baseline",
                            "patients",
                            "patient={}".format(patient),
                        )
                        Folder.create(patient_dir)

                    # for idl gtvn, only create folder for test set
                    if inference_type == "idl_gtvn" and dataset == "test":
                        patient_dir = os.path.join(
                            epoch_dir,
                            "patients",
                            "patient={}".format(patient),
                        )
                        Folder.create(patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self._patient_inference(
                        patient=patient,
                        hyper=hyper,
                        inference_type=inference_type,
                        idl_gtvn_baseline_epoch_dir=idl_gtvn_baseline_epoch_dir,
                    )

                    # save preds of current patient
                    if inference_type == "baseline":
                        if dataset == "test":
                            gtv_list = ["gtvt", "gtvn"]
                        else:
                            gtv_list = ["gtvn"]
                    else:
                        if dataset == "test":
                            gtv_list = ["gtvn"]
                            # save clicks.nii
                            Nii.save(
                                img=patient_results["gtvn"]["clicks"],
                                path=os.path.join(patient_dir, "distance_map.nii"),
                                spacing=g.NII_SPACING,
                            )
                        else:
                            gtv_list = []
                    for gtv in gtv_list:
                        Nii.save(
                            img=patient_results[gtv]["pred"],
                            path=os.path.join(patient_dir, "pred_{}.nii".format(gtv)),
                            spacing=g.NII_SPACING,
                        )

                    # record score of current patient
                    if inference_type == "baseline":
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                # save cur patient score into inference_test.json (test set only)
                                if dataset == "test":
                                    epoch_scores["patient={}".format(patient)][gtv][
                                        metric
                                    ] = patient_results[gtv][metric]
                                # add scores of current patient into median(list)
                                if dataset == "test" or dataset == "valid":
                                    epoch_scores["median"][gtv][metric].append(
                                        patient_results[gtv][metric]
                                    )
                    else:
                        for metric in g.METRICS:
                            if dataset == "test":
                                # save cur patient score into inference_test.json (test set only)
                                epoch_scores["patient={}".format(patient)][metric][
                                    "round=01"
                                ] = patient_results["gtvn"][metric]
                                # add scores of current patient into median(list)
                                # record in ["round=01"] for test set
                                epoch_scores["median"][metric]["round=01"].append(
                                    patient_results["gtvn"][metric]
                                )
                            # add scores of current patient into median(list)
                            if dataset == "valid":
                                epoch_scores["median"][metric].append(
                                    patient_results["gtvn"][metric]
                                )

                # all patients under current epoch have been traversed
                # no need to calculate median score on training set
                if dataset == "train":
                    continue  # next epoch dir

                if dataset == "test" or dataset == "valid":
                    # calculate median score (test and valid set only)
                    if inference_type == "baseline":
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                median = epoch_scores["median"][gtv][metric]
                                epoch_scores["median"][gtv][metric] = statistics.median(
                                    median
                                )
                    else:
                        for metric in g.METRICS:
                            if dataset == "test":
                                median = epoch_scores["median"][metric]["round=01"]
                                epoch_scores["median"][metric][
                                    "round=01"
                                ] = statistics.median(median)
                            elif dataset == "valid":
                                median = epoch_scores["median"][metric]
                                epoch_scores["median"][metric] = statistics.median(
                                    median
                                )
                    # save all patients scores in "inference_test.json"
                    if inference_type == "baseline":
                        json_save_path = os.path.join(
                            epoch_dir, "baseline", "inference_{}.json".format(dataset)
                        )
                    else:
                        json_save_path = os.path.join(
                            epoch_dir, "inference_{}.json".format(dataset)
                        )
                    Json.save(
                        data=epoch_scores,
                        path=json_save_path,
                    )
                    continue  # next epoch dir

    def remove_non_optimal_epochs(self, baseline_id: str, dataset: str = "valid"):
        self._remove_non_optimal_epochs(train_id=baseline_id, dataset=dataset)

    def _remove_non_optimal_epochs(self, train_id: str, dataset: str):
        # record top median scores through each fold and epoch
        top_scores_set = Dict()

        if dataset != "valid":
            dataset = "test"

        train_result_dir = self._find_result_dir(train_id)

        for fold_dir in Explorer.get_sub_folders(
            train_result_dir, key_word="fold=", return_full_path=True
        ):
            fold = Path(fold_dir).name

            for epoch_dir in Explorer.get_sub_folders(
                fold_dir, key_word="epoch=", return_full_path=True
            ):
                epoch = Path(epoch_dir).name

                # load scores of current epoch
                # if baseline
                if "baseline_" in train_id:
                    epoch_scores = Json.load(
                        os.path.join(
                            epoch_dir,
                            "baseline",
                            "inference_{}.json".format(dataset),
                        )
                    )["median"]["gtvs"]
                else:
                    epoch_scores = Json.load(
                        os.path.join(epoch_dir, "inference_{}.json".format(dataset))
                    )["median"]
                    if dataset == "test":
                        for metric in g.METRICS:
                            epoch_scores[metric] = epoch_scores[metric]["round=01"]

                # delete nan msd/hd95 results
                if math.isnan(epoch_scores["msd"]) or math.isnan(epoch_scores["hd95"]):
                    print("delete:", epoch_dir)
                    Folder.delete(epoch_dir)
                    continue  # goto next epoch dir

                # keep cur epoch scores if top scores set is empty
                if top_scores_set == {}:
                    top_scores_set[train_id][fold][epoch] = epoch_scores
                    continue  # goto next epoch dir
                else:
                    top_scores_set = self.__walk_top_scores_set(
                        top_scores_set=top_scores_set,
                        epoch_scores=epoch_scores,
                        epoch_dir=epoch_dir,
                    )

    # a sub function of _remove_non_optimal_epochs()
    def __walk_top_scores_set(
        self,
        top_scores_set: Dict,
        epoch_scores: Dict,
        epoch_dir: str,
    ):
        train_results_dir = Path(epoch_dir).parent.parent.parent

        delete_epoch_dir = False
        add_epoch_scores = False

        # copy dict to avoid conflict between for loop and pop
        top_scores_set_copy = top_scores_set.copy()

        # loop through top_scores_set
        for train_id in top_scores_set.keys():
            for fold in top_scores_set[train_id].keys():
                for epoch in top_scores_set[train_id][fold].keys():

                    top_scores = top_scores_set[train_id][fold][epoch]

                    # delete epoch_dir,
                    # if it is worse than anyone in top_scores_set
                    if (
                        epoch_scores["dsc"] < top_scores["dsc"]
                        and epoch_scores["msd"] > top_scores["msd"]
                        and epoch_scores["hd95"] > top_scores["hd95"]
                    ):
                        delete_epoch_dir = True

                    # add epoch_scores in to top_scores_set,
                    # if it is comparable for one of the top_scores_set
                    # (comparable means: at least one of dsc/msd/hd95 is better)
                    if (
                        epoch_scores["dsc"] > top_scores["dsc"]
                        or epoch_scores["msd"] < top_scores["msd"]
                        or epoch_scores["hd95"] < top_scores["hd95"]
                    ):
                        add_epoch_scores = True

                    # delete anyone in top_scores_set,
                    # if it is worse than epoch_scores
                    if (
                        top_scores["dsc"] < epoch_scores["dsc"]
                        and top_scores["msd"] > epoch_scores["msd"]
                        and top_scores["hd95"] > epoch_scores["hd95"]
                    ):
                        delete_dir = os.path.join(
                            train_results_dir, train_id, fold, epoch
                        )
                        print("delete:", delete_dir)
                        Folder.delete(delete_dir)

                        # remove from dict
                        top_scores_set_copy[train_id][fold].pop(epoch)
                        if top_scores_set_copy[train_id][fold] == {}:
                            top_scores_set_copy[train_id].pop(fold)
                        if top_scores_set_copy[train_id] == {}:
                            top_scores_set_copy.pop(train_id)

        # copy data back
        top_scores_set = top_scores_set_copy

        # delete epoch dir
        if delete_epoch_dir:
            print("delete:", epoch_dir)
            Folder.delete(epoch_dir)

        # add epoch scores into top scores set
        if not delete_epoch_dir and add_epoch_scores:
            epoch = Path(epoch_dir).name
            fold = Path(epoch_dir).parent.name
            train_id = Path(epoch_dir).parent.parent.name
            top_scores_set[train_id][fold][epoch] = epoch_scores

        return top_scores_set
