import global_elems as g
import os
import torch
import math
import statistics
from tqdm import tqdm
from datetime import datetime
from custom import Dict
from torch.utils.data import DataLoader
from shared_training import SharedTraining
from matplotlib import pyplot as plt
from baseline_dataset import BaselineDataSet
from loss_func import UnifiedFocalLoss


class BaselineTraining(SharedTraining):
    def __load_dataset(self, fold: int, debug_mode: bool = False):
        json_data = g.load_json(g.DATASET_SPLIT_JSON)

        if len(json_data) - 1 != g.DATASET_K_FOLDS:
            json_data = super()._split_dataset()

        test_patients = g.str_to_list(json_data["test.set"])
        valid_patients = g.str_to_list(json_data["fold.{}".format(fold)])

        train_patients = []
        for i in json_data:
            if i != "test.set" and i != "fold.{}".format(fold):
                train_patients += g.str_to_list(json_data[i])

        if debug_mode:
            batch_size = g.MAX_BATCH_SIZE_PER_GPU
            if g.used_gpu_count() > 1:
                batch_size *= g.used_gpu_count()

            train_patients = train_patients[:batch_size]
            valid_patients = valid_patients[:batch_size]
            test_patients = test_patients[:batch_size]

        return train_patients, valid_patients, test_patients

    def _load_hyper(
        self,
        hyper: Dict,
        fold: int,
        cnn_path: str = "",  # make cnn_path == "" will load a new cnn
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # Change int/float/str/list into Int/Float/Str/List
        hyper.convert_data_type()

        # cross valid folds
        hyper["dataset"]["k.folds"] = g.DATASET_K_FOLDS

        # use cross validation or not
        hyper["dataset"]["cross.valid"] = hyper["cross.valid"]
        hyper.pop("cross.valid")

        # at least 2 epochs to compare loss difference
        if debug_mode:
            hyper["epoch"]["init"] = 2
        else:
            hyper["epoch"]["init"].set_range(1, None)

        # record actual epochs because of early stop
        hyper["epoch"]["actual"] = 0

        # early stop, based on epoch
        hyper["epoch"]["early.stop"].set_range(1, hyper["epoch"]["init"])

        # lr
        hyper["lr"]["init"].set_range(g.EPS, 1)

        # actual lr
        hyper["lr"]["actual"] = hyper["lr"]["init"]
        if g.used_gpu_count() > 1:
            hyper["lr"]["actual"] *= g.used_gpu_count()

        # min lr
        hyper["lr"]["min"].set_range(g.EPS, hyper["lr"]["init"])

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        hyper["lr"]["decay.patience"].set_range(1, hyper["epoch"]["init"])

        # number of best valid loss cnn retained
        hyper["keep.best.cnn.num"].set_range(1, hyper["epoch"]["init"])

        # augment percent
        hyper["augment"]["pct"].set_range(0, 1)

        # load shared hyper parameters
        super()._load_hyper(hyper=hyper, cnn_path=cnn_path)

        # run this after shared hyper loaded, because loss parameters are needed
        hyper["loss"]["func"] = UnifiedFocalLoss(
            asym=hyper["loss"]["asym"],
            weight=hyper["loss"]["weight"],
            delta=hyper["loss"]["delta"],
            gamma=hyper["loss"]["gamma"],
            gtvt_only=False,
        ).to(g.DEVICE)

        # load patients
        (
            train_patients,
            valid_patients,
            test_patients,
        ) = self.__load_dataset(fold=fold, debug_mode=debug_mode)

        hyper["dataset"]["train.len"] = train_patients.__len__()
        hyper["dataset"]["valid.len"] = valid_patients.__len__()
        hyper["dataset"]["test.len"] = test_patients.__len__()
        hyper["dataset"]["len"] = (
            train_patients.__len__()
            + valid_patients.__len__()
            + test_patients.__len__()
        )

        # create datasets
        # run this after shared hyper loaded, because hyper["augment"] is needed
        train_set = BaselineDataSet(
            patient_list=train_patients, augment=hyper["augment"]
        )
        valid_set = BaselineDataSet(patient_list=valid_patients)
        test_set = BaselineDataSet(patient_list=test_patients)

        # dataloader
        hyper["train.loader"] = DataLoader(
            dataset=train_set,
            batch_size=hyper["batch.size"],
            shuffle=True,  # only shuffle train loader
            num_workers=g.NUM_WORKERS,
        )
        hyper["valid.loader"] = DataLoader(
            dataset=valid_set,
            batch_size=hyper["batch.size"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )
        hyper["test.loader"] = DataLoader(
            dataset=test_set,
            batch_size=hyper["batch.size"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )

    def __get_simple_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for i in hyper:
            if i == "train.loader" or i == "valid.loader" or i == "test.loader":
                pass

            elif isinstance(hyper[i], list) or isinstance(hyper[i], dict):
                simple_hyper[i] = hyper[i].copy()
            else:
                simple_hyper[i] = hyper[i]

        return simple_hyper

    def __print_hyper(self, hyper: Dict):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def __save_hyper(self, hyper: Dict, json_path: str):
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

    def __training(self, hyper: Dict, cur_fold_folder: str):
        best_loss_dict = Dict()
        loss_save_path = os.path.join(cur_fold_folder, "hyper", "loss.json")
        lr_save_path = os.path.join(cur_fold_folder, "hyper", "lr.json")
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
            cur_epoch_loss = Dict()
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
                g.create_folder(os.path.join(cur_epoch_folder, "baseline"))
                cnn_save_path = os.path.join(
                    cur_epoch_folder,
                    "baseline",
                    "epoch={:03d}.pt".format(cur_epoch),
                )
                self._save_cnn(hyper, cnn_save_path)
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
                    g.create_folder(os.path.join(cur_epoch_folder, "baseline"))
                    cnn_save_path = os.path.join(
                        cur_epoch_folder,
                        "baseline",
                        "epoch={:03d}.pt".format(cur_epoch),
                    )
                    self._save_cnn(hyper, cnn_save_path)
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
        for cur_hyper in self._load_group_hyper(g.HYPER_JSON_BASELINE):

            baseline_id = "baseline_" + self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.HYPER_JSON_BASELINE,
                hyper=cur_hyper,
                debug_mode=debug_mode,
            )
            g.print_line()
            print(baseline_id)

            baseline_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id)
            g.create_folder(baseline_folder)

            for cur_fold in range(1, g.DATASET_K_FOLDS + 1):
                cur_fold_folder = os.path.join(
                    baseline_folder, "fold={:02d}".format(cur_fold)
                )
                g.create_folder(cur_fold_folder)

                self._load_hyper(
                    hyper=cur_hyper,
                    fold=cur_fold,
                    cnn_path="",  # cnn_path="" will create a new cnn
                    debug_mode=debug_mode,
                )
                if cur_fold == 1:
                    g.print_line()
                    self.__print_hyper(cur_hyper)

                g.print_line()
                print("cross validation fold: {}".format(cur_fold))

                cur_hyper_folder = os.path.join(cur_fold_folder, "hyper")
                g.create_folder(cur_hyper_folder)
                # save an empty loss.json
                g.save_json(Dict(), os.path.join(cur_hyper_folder, "loss.json"))
                # save an empty lr.json
                g.save_json(Dict(), os.path.join(cur_hyper_folder, "lr.json"))

                # save hyper before training
                hyper_save_path = os.path.join(cur_hyper_folder, "hyper.json")
                self.__save_hyper(cur_hyper, hyper_save_path)

                # start training
                cur_hyper["time.spent"] = datetime.now()
                self.__training(cur_hyper, cur_fold_folder)
                cur_hyper["time.spent"] = datetime.now() - cur_hyper["time.spent"]
                cur_hyper["time.spent"] = str(cur_hyper["time.spent"]).split(".", 2)[0]

                # save hyper after training
                self.__save_hyper(cur_hyper, hyper_save_path)

                # clear time spent before next training
                cur_hyper.pop("time.spent")

                # stop if no cross validation
                if cur_hyper["dataset"]["cross.valid"] is False:
                    break

            # inference
            for dataset in ["valid", "test"]:
                self.inference(
                    baseline_id=baseline_id,
                    dataset=dataset,
                    debug_mode=debug_mode,
                )

    def inference(
        self,
        baseline_id: str,
        dataset: str = "test",
        debug_mode: bool = False,
    ):
        g.print_line()
        print("inference on {} set: {}".format(dataset, baseline_id))

        baseline_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id)

        if dataset == "valid":
            best_scores = Dict()

        # loop through fold folders
        for cur_fold_folder in g.get_sub_folders(baseline_folder, key_word="fold="):
            cur_fold = int(cur_fold_folder[len("fold=") :])
            g.print_line()
            print("current fold: ", cur_fold)
            cur_fold_folder = os.path.join(
                g.TRAIN_RESULTS_FOLDER, baseline_id, cur_fold_folder
            )

            # loop through epoch folders
            for cur_epoch_folder in g.get_sub_folders(
                cur_fold_folder, key_word="epoch="
            ):
                cur_epoch = int(cur_epoch_folder[len("epoch=") :])
                print("current epoch: ", cur_epoch)
                cur_epoch_folder = os.path.join(cur_fold_folder, cur_epoch_folder)

                # initialize median score
                cur_score = Dict()
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in g.METRICS:
                        cur_score["median"][gtv][metric] = []

                # load cnn
                cur_cnn_path = g.get_sub_files(
                    os.path.join(cur_epoch_folder, "baseline"),
                    key_word=".pt",
                    return_full_path=True,
                )[0]
                # create an empty hyper dict to save cnn
                cur_hyper = Dict()
                self._load_cnn(hyper=cur_hyper, cnn_path=cur_cnn_path)

                # load dataset
                patient_list = self.__load_dataset(fold=cur_fold, debug_mode=debug_mode)
                if dataset == "test":
                    patient_list = patient_list[2]
                else:
                    patient_list = patient_list[1]

                for cur_patient in tqdm(patient_list):
                    # if testset, create folder to save cur patient preds
                    if dataset == "test":
                        cur_patient_folder = os.path.join(
                            cur_epoch_folder,
                            "baseline",
                            "patients",
                            "patient={}".format(cur_patient),
                        )
                        g.create_folder(cur_patient_folder)

                    # result structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    cur_patient_result = self._inference_single_patient(
                        patient=cur_patient,
                        hyper=cur_hyper,
                        gtvt_only=False,
                    )

                    # save score of cur patient
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric in g.METRICS:
                            # add cur patient result in avg_list
                            cur_score["median"][gtv][metric].append(
                                cur_patient_result[gtv][metric]
                            )
                            # record cur patient result if on test test
                            if dataset == "test":
                                cur_score["patient={}".format(cur_patient)][gtv][
                                    metric
                                ] = cur_patient_result[gtv][metric]

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

                # calculate median score
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in g.METRICS:
                        median = cur_score["median"][gtv][metric]
                        cur_score["median"][gtv][metric] = statistics.median(median)

                # save score (test set)
                if dataset == "test":
                    g.save_json(
                        data=cur_score,
                        path=os.path.join(
                            cur_epoch_folder, "baseline", "score_test.json"
                        ),
                    )
                    continue

                # valid set, delete non-optimal folds and epochs
                if math.isnan(cur_score["median"]["gtvs"]["msd"]) or math.isnan(
                    cur_score["median"]["gtvs"]["hd95"]
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
                                cur_score["median"]["gtvs"]["dsc"]
                                < best_scores[fold][epoch]["dsc"]
                                and cur_score["median"]["gtvs"]["msd"]
                                > best_scores[fold][epoch]["msd"]
                                and cur_score["median"]["gtvs"]["hd95"]
                                > best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = False
                                break
                            # cur_score is better than best_score, save
                            # (at least one of dsc/msd/hd95 is better)
                            if (
                                cur_score["median"]["gtvs"]["dsc"]
                                > best_scores[fold][epoch]["dsc"]
                                or cur_score["median"]["gtvs"]["msd"]
                                < best_scores[fold][epoch]["msd"]
                                or cur_score["median"]["gtvs"]["hd95"]
                                < best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = True
                            # best_score is worse than cur score, delete best_score
                            if (
                                best_scores[fold][epoch]["dsc"]
                                < cur_score["median"]["gtvs"]["dsc"]
                                and best_scores[fold][epoch]["msd"]
                                > cur_score["median"]["gtvs"]["msd"]
                                and best_scores[fold][epoch]["hd95"]
                                > cur_score["median"]["gtvs"]["hd95"]
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
                    best_scores[cur_fold][cur_epoch] = cur_score["median"]["gtvs"]
                    g.save_json(
                        data=cur_score,
                        path=os.path.join(
                            cur_epoch_folder, "baseline", "score_valid.json"
                        ),
                    )
                else:
                    g.delete_folder(cur_epoch_folder)
