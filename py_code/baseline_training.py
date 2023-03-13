import global_elems as g
import os
import torch
import math
from tqdm import tqdm
from datetime import datetime
from nested_dict import NestedDict
from torch.utils.data import DataLoader
from shared_training import SharedTraining
from matplotlib import pyplot as plt
from baseline_dataset import BaselineDataSet
from loss_func import UnifiedFocalLoss


class BaselineTraining(SharedTraining):
    def __load_dataset(self, hyper: NestedDict, fold: int, debug_mode: bool = False):
        json_data = g.load_json(g.DATASET_SPLITTING_JSON)

        if len(json_data) - 1 != g.DATASET_K_FOLDS:
            json_data = super()._split_dataset()

        test_patients = g.str_to_list(json_data["test.patients"])
        valid_patients = g.str_to_list(json_data["fold.{}".format(fold)])

        train_patients = []
        for i in json_data:
            if i != "test.patients" and i != "fold.{}".format(fold):
                train_patients += g.str_to_list(json_data[i])

        # if 1:
        #     print(set(train_patients) & set(valid_patients))

        if debug_mode:
            train_patients = train_patients[: hyper["batch.size"]["actual"]]
            valid_patients = valid_patients[: hyper["batch.size"]["actual"]]
            test_patients = test_patients[: hyper["batch.size"]["actual"]]

        return train_patients, valid_patients, test_patients

    def __load_hyper(
        self,
        hyper: NestedDict,
        fold: int,
        exist_cnn_path: str = "",
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # cross valid folds
        hyper["dataset"]["k.folds"] = g.DATASET_K_FOLDS

        # use cross validation or not
        hyper["dataset"]["cross.valid"] = bool(hyper["cross.valid"])
        hyper.pop("cross.valid")

        # epochs
        if debug_mode:
            # at least 2 epochs to compare difference in loss
            hyper["epoch"]["init"] = 2
        else:
            hyper["epoch"]["init"] = int(hyper["epoch"]["init"])
            hyper["epoch"]["init"] = g.check_limit(hyper["epoch"]["init"], 1, None)

        # record actual epochs because of early stop
        hyper["epoch"]["actual"] = 0

        # early stop, based on epoch
        hyper["epoch"]["early.stop"] = int(hyper["epoch"]["early.stop"])
        hyper["epoch"]["early.stop"] = g.check_limit(
            hyper["epoch"]["early.stop"], 1, hyper["epoch"]["init"]
        )

        # lr
        hyper["lr"]["init"] = float(hyper["lr"]["init"])
        hyper["lr"]["init"] = g.check_limit(hyper["lr"]["init"], 1e-10, None)

        # actual lr
        if g.used_gpu_count() > 1:
            hyper["lr"]["actual"] = hyper["lr"]["init"] * g.used_gpu_count()
        else:
            hyper["lr"]["actual"] = hyper["lr"]["init"]

        # min lr
        hyper["lr"]["min"] = float(hyper["lr"]["min"])
        hyper["lr"]["min"] = g.check_limit(hyper["lr"]["min"], 0, hyper["lr"]["init"])

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        hyper["lr"]["decay.patience"] = int(hyper["lr"]["decay.patience"])
        hyper["lr"]["decay.patience"] = g.check_limit(
            hyper["lr"]["decay.patience"], 1, hyper["epoch"]["init"]
        )

        # number of best valid loss cnn retained
        hyper["keep.best.cnn.num"] = int(hyper["keep.best.cnn.num"])
        hyper["keep.best.cnn.num"] = g.check_limit(
            hyper["keep.best.cnn.num"], 1, hyper["epoch"]["init"]
        )

        # augmentation percent
        hyper["augment"]["pct"] = float(hyper["augment"]["pct"])
        hyper["augment"]["pct"] = g.check_limit(hyper["augment"]["pct"], 0, 1)

        # load shared hyper parameters
        super()._load_hyper(
            hyper=hyper,
            exist_cnn_path=exist_cnn_path,
        )

        # run this after shared hyper loaded, loss parameters are needed
        hyper["loss"]["func"] = UnifiedFocalLoss(
            asym=hyper["loss"]["asym"],
            weight=hyper["loss"]["weight"],
            delta=hyper["loss"]["delta"],
            gamma=hyper["loss"]["gamma"],
            gtvt_only=False,
        ).to(g.DEVICE)

        # run this after shared hyper loaded, actual batch size is needed
        (
            train_patients,
            valid_patients,
            test_patients,
        ) = self.__load_dataset(hyper=hyper, fold=fold, debug_mode=debug_mode)

        hyper["dataset"]["train.len"] = train_patients.__len__()
        hyper["dataset"]["valid.len"] = valid_patients.__len__()
        hyper["dataset"]["test.len"] = test_patients.__len__()
        hyper["dataset"]["len"] = (
            train_patients.__len__()
            + valid_patients.__len__()
            + test_patients.__len__()
        )

        # create dataset
        train_set = BaselineDataSet(
            patient_list=train_patients, augment=hyper["augment"]
        )
        valid_set = BaselineDataSet(patient_list=valid_patients)
        test_set = BaselineDataSet(patient_list=test_patients)

        # dataloader
        hyper["train.loader"] = DataLoader(
            dataset=train_set,
            batch_size=hyper["batch.size"]["actual"],
            shuffle=True,  # only shuffle train loader
            num_workers=g.NUM_WORKERS,
        )
        hyper["valid.loader"] = DataLoader(
            dataset=valid_set,
            batch_size=hyper["batch.size"]["actual"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )
        hyper["test.loader"] = DataLoader(
            dataset=test_set,
            batch_size=hyper["batch.size"]["actual"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )

    def __get_simple_hyper(self, hyper: NestedDict) -> NestedDict:
        simple_hyper = NestedDict()
        for i in hyper:
            if i == "train.loader" or i == "valid.loader" or i == "test.loader":
                pass

            elif isinstance(hyper[i], list) or isinstance(hyper[i], dict):
                simple_hyper[i] = hyper[i].copy()
            else:
                simple_hyper[i] = hyper[i]

        return simple_hyper

    def __print_hyper(self, hyper: NestedDict):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def __save_hyper(self, hyper: NestedDict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._save_hyper(simple_hyper, json_path)

    def draw_lr_fig(self, baseline_id: str):
        lr_json_path = os.path.join(
            g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline", "lr.json"
        )
        self.__draw_lr_fig(lr_json_path)

    def __draw_lr_fig(self, lr_json_path: str):
        plt.figure().clear()

        lr_dict = g.load_json(lr_json_path)
        lr_list = []
        for i in lr_dict:
            lr_list.append(lr_dict[i])

        plt.plot(range(1, len(lr_list) + 1), lr_list, label="lr")
        plt.legend()
        plt.savefig(lr_json_path[:-4] + "png")

    def draw_loss_fig(self, baseline_id: str):
        loss_json_path = os.path.join(
            g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline", "loss.json"
        )
        self.__draw_loss_fig(loss_json_path)

    def __draw_loss_fig(self, loss_json_path: str):
        loss_dict = g.load_json(loss_json_path)
        train_loss = []
        valid_loss = []

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

    def __training(self, hyper: NestedDict, cur_fold_folder: str):
        best_loss_dict = NestedDict()
        loss_save_path = os.path.join(cur_fold_folder, "loss.json")
        lr_save_path = os.path.join(cur_fold_folder, "lr.json")
        patience_count = 0

        for cur_epoch in range(1, hyper["epoch"]["init"] + 1):
            g.print_line()
            print("epoch: {}".format(cur_epoch))
            print("training:")
            hyper["cnn"].train()
            sum_loss = 0
            batch_num = 0
            for inputs, labels in tqdm(hyper["train.loader"]):
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = hyper["cnn"](inputs)[3]
                loss = hyper["loss"]["func"](outputs, labels, weight_map=None)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                sum_loss += loss.item()
                batch_num += 1
            train_loss = sum_loss / batch_num

            # validation
            print("validation:")
            hyper["cnn"].eval()
            with torch.no_grad():
                sum_loss = 0
                batch_num = 0
                for inputs, labels in tqdm(hyper["valid.loader"]):
                    inputs = inputs.to(g.DEVICE)
                    labels = labels.to(g.DEVICE)
                    outputs = hyper["cnn"](inputs)[3]
                    loss = hyper["loss"]["func"](outputs, labels, weight_map=None)
                    sum_loss += loss.item()
                    batch_num += 1
            valid_loss = sum_loss / batch_num
            hyper["scheduler"].step(valid_loss)

            # current epoch finished
            hyper["epoch"]["actual"] = cur_epoch

            # save loss in json
            loss_dict = g.load_json(loss_save_path)
            cur_epoch_loss = NestedDict()
            cur_epoch_loss["train"] = train_loss
            cur_epoch_loss["valid"] = valid_loss
            loss_dict["epoch={:03d}".format(hyper["epoch"]["actual"])] = cur_epoch_loss
            g.save_json(loss_dict, loss_save_path)
            # draw loss figure
            self.__draw_loss_fig(loss_save_path)

            # save lr in json
            lr_dict = g.load_json(lr_save_path)
            for param_group in hyper["optim"].param_groups:
                cur_epoch_lr = param_group["lr"]
            lr_dict["epoch={:03d}".format(hyper["epoch"]["actual"])] = cur_epoch_lr
            g.save_json(lr_dict, lr_save_path)
            # draw lr figure
            self.__draw_lr_fig(lr_save_path)

            # save cnn
            if len(best_loss_dict) < hyper["keep.best.cnn.num"]:
                best_loss_dict[cur_epoch] = valid_loss
                cur_epoch_folder = os.path.join(
                    cur_fold_folder, "epoch={:03d}".format(cur_epoch)
                )
                g.create_folder(cur_epoch_folder)
                self._save_cnn(hyper, os.path.join(cur_epoch_folder, "cnn.pt"))
            else:
                worst_epoch = g.get_dict_key_max(best_loss_dict)
                worst_loss = best_loss_dict[worst_epoch]
                if valid_loss < worst_loss:
                    g.delete_folder(
                        os.path.join(
                            cur_fold_folder, "epoch={:03d}".format(worst_epoch)
                        )
                    )
                    best_loss_dict.pop(worst_epoch)
                    best_loss_dict[cur_epoch] = valid_loss
                    cur_epoch_folder = os.path.join(
                        cur_fold_folder, "epoch={:03d}".format(cur_epoch)
                    )
                    g.create_folder(cur_epoch_folder)
                    self._save_cnn(hyper, os.path.join(cur_epoch_folder, "cnn.pt"))
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= hyper["epoch"]["early.stop"]:
                        break

    def training(
        self,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        for hyper in self._load_group_hyper(g.HYPER_JSON_BASELINE):

            baseline_id = "baseline_" + self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.HYPER_JSON_BASELINE,
                hyper=hyper,
                debug_mode=debug_mode,
            )
            g.print_line()
            print(baseline_id)

            baseline_folder = os.path.join(
                g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline"
            )
            g.create_folder(baseline_folder)

            for cur_fold in range(1, g.DATASET_K_FOLDS + 1):
                cur_fold_folder = os.path.join(
                    baseline_folder, "fold={:02d}".format(cur_fold)
                )
                g.create_folder(cur_fold_folder)

                self.__load_hyper(
                    hyper=hyper,
                    fold=cur_fold,
                    exist_cnn_path="",
                    debug_mode=debug_mode,
                )
                if cur_fold == 1:
                    g.print_line()
                    self.__print_hyper(hyper)

                g.print_line()
                print("cross validation fold: {}".format(cur_fold))

                # save an empty loss.json
                g.save_json(NestedDict(), os.path.join(cur_fold_folder, "loss.json"))
                # save an empty lr.json
                g.save_json(NestedDict(), os.path.join(cur_fold_folder, "lr.json"))

                # save hyper before training
                hyper_save_path = os.path.join(cur_fold_folder, "hyper.json")
                self.__save_hyper(hyper, hyper_save_path)

                # start training
                hyper["time.used"] = datetime.now()
                self.__training(hyper, cur_fold_folder)
                hyper["time.used"] = datetime.now() - hyper["time.used"]
                hyper["time.used"] = str(hyper["time.used"]).split(".", 2)[0]

                # save hyper after training
                self.__save_hyper(hyper, hyper_save_path)

                # clear time_used before next training
                hyper["time.used"] = None

                if hyper["dataset"]["cross.valid"] is False:
                    break

            # inference
            for dataset in ["valid", "test"]:
                self.inference(
                    baseline_id=baseline_id,
                    dataset=dataset,
                    print_hyper=False,
                    debug_mode=debug_mode,
                )

    def inference(
        self,
        baseline_id: str,
        dataset: str,
        print_hyper: bool = True,
        debug_mode: bool = False,
    ):
        g.print_line()
        print("inference on {} set: {}".format(dataset, baseline_id))

        baseline_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline")

        if dataset == "valid":
            best_scores = NestedDict()

        for cur_fold_folder in g.get_sub_folders(baseline_folder, key_word="fold"):

            cur_fold = int(cur_fold_folder[len("fold=") :])
            g.print_line()
            print("current fold: ", cur_fold)
            cur_fold_folder = os.path.join(
                g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline", cur_fold_folder
            )
            hyper = g.load_json(os.path.join(cur_fold_folder, "hyper.json"))

            for cur_epoch_folder in g.get_sub_folders(
                cur_fold_folder, key_word="epoch"
            ):

                cur_epoch = int(cur_epoch_folder[len("epoch=") :])
                print("current epoch: ", cur_epoch)
                cur_epoch_folder = os.path.join(cur_fold_folder, cur_epoch_folder)

                # load and print hyper
                self.__load_hyper(
                    hyper=hyper,
                    fold=cur_fold,
                    exist_cnn_path=os.path.join(cur_epoch_folder, "cnn.pt"),
                    debug_mode=debug_mode,
                )
                if print_hyper:
                    g.print_line()
                    self.__print_hyper(hyper)
                    g.print_line()

                cur_score = NestedDict()
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric_type in g.METRICS_LIST:
                        cur_score["avg"][gtv][metric_type] = []

                if dataset == "test":
                    patient_list = hyper["test.loader"].dataset.patient_list
                else:
                    patient_list = hyper["valid.loader"].dataset.patient_list

                for cur_patient in tqdm(patient_list):
                    # if testset, create folder to save cur patient preds
                    if dataset == "test":
                        cur_patient_folder = os.path.join(
                            cur_epoch_folder,
                            "patients",
                            "patient={}".format(cur_patient),
                        )
                        g.create_folder(cur_patient_folder)

                    # result structure: gtvs/gtvt/gvtn → pred/dsc/msd/hd95
                    cur_patient_result = self._inference_single_patient(
                        patient=cur_patient, hyper=hyper
                    )

                    # save score of cur patient
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric_type in g.METRICS_LIST:
                            # add cur patient result in avg_list
                            cur_score["avg"][gtv][metric_type].append(
                                cur_patient_result[gtv][metric_type]
                            )
                            # record cur patient result if on test test
                            if dataset == "test":
                                cur_score["patient={}".format(cur_patient)][gtv][
                                    metric_type
                                ] = cur_patient_result[gtv][metric_type]

                    # save pred of cur patient
                    if dataset == "test":
                        for gtv in ["gtvt", "gtvn", "gtvs"]:
                            g.save_nii(
                                img=cur_patient_result[gtv]["pred"],
                                save_path=os.path.join(
                                    cur_patient_folder, "pred_{}.nii".format(gtv)
                                ),
                                spacing=g.NII_SPACING,
                            )
                            g.save_nii(
                                img=g.binarize_img(cur_patient_result[gtv]["pred"]),
                                save_path=os.path.join(
                                    cur_patient_folder, "pred_{}_binary.nii".format(gtv)
                                ),
                                spacing=g.NII_SPACING,
                            )

                # get avg score
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric_type in g.METRICS_LIST:
                        avg = cur_score["avg"][gtv][metric_type]
                        cur_score["avg"][gtv][metric_type] = sum(avg) / len(avg)

                # save score (test set)
                if dataset == "test":
                    g.save_json(
                        data=cur_score,
                        path=os.path.join(cur_epoch_folder, "score_test.json"),
                    )
                    continue

                # valid set, delete non-optimal folds and epochs
                if math.isnan(cur_score["avg"]["gtvs"]["msd"]) or math.isnan(
                    cur_score["avg"]["gtvs"]["hd95"]
                ):
                    g.delete_folder(cur_epoch_folder)
                    continue

                if len(best_scores) == 0:
                    save_cur_score = True
                else:
                    save_cur_score = None
                    # loop through best_scores
                    for fold in g.get_dict_keys(best_scores):
                        for epoch in g.get_dict_keys(best_scores[fold]):
                            # cur_score is worse than best_score, dont save
                            if (
                                cur_score["avg"]["gtvs"]["dsc"]
                                < best_scores[fold][epoch]["dsc"]
                                and cur_score["avg"]["gtvs"]["msd"]
                                > best_scores[fold][epoch]["msd"]
                                and cur_score["avg"]["gtvs"]["hd95"]
                                > best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = False
                                break
                            # cur_score is better than best_score, save
                            # (at least one of dsc/msd/hd95 is better)
                            if (
                                cur_score["avg"]["gtvs"]["dsc"]
                                > best_scores[fold][epoch]["dsc"]
                                or cur_score["avg"]["gtvs"]["msd"]
                                < best_scores[fold][epoch]["msd"]
                                or cur_score["avg"]["gtvs"]["hd95"]
                                < best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = True
                            # best_score is worse than cur score, delete best_score
                            if (
                                best_scores[fold][epoch]["dsc"]
                                < cur_score["avg"]["gtvs"]["dsc"]
                                and best_scores[fold][epoch]["msd"]
                                > cur_score["avg"]["gtvs"]["msd"]
                                and best_scores[fold][epoch]["hd95"]
                                > cur_score["avg"]["gtvs"]["hd95"]
                            ):
                                g.delete_folder(
                                    os.path.join(
                                        baseline_folder,
                                        "fold={:02d}".format(fold),
                                        "epoch={:03d}".format(epoch),
                                    )
                                )
                        if save_cur_score is False:
                            break

                # save cur avg score
                if save_cur_score:
                    best_scores[cur_fold][cur_epoch] = cur_score["avg"]["gtvs"]
                    g.save_json(
                        data=cur_score,
                        path=os.path.join(cur_epoch_folder, "score_valid.json"),
                    )
                else:
                    g.delete_folder(cur_epoch_folder)
