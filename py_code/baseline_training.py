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


class BaselineTraining(SharedTraining):
    def __init__(self):
        super().__init__()

        # record self._epochs_actual because of "early.stop.patience"
        self._epochs_actual = 0

    def _load_hyper(
        self,
        hyper: dict,
        fold: int,
        exist_cnn_path: str = None,
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # epochs
        if debug_mode:
            self._epochs = 2  # run 2 epochs to compare difference in loss
        else:
            self._epochs = int(hyper["epochs"])
            self._epochs = g.check_limit(self._epochs, 1, None)

        # lr
        self._lr = float(hyper["lr"])
        self._lr = g.check_limit(self._lr, 1e-10, None)
        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._lr_actual = self._lr * used_gpu_count
        else:
            self._lr_actual = self._lr

        # min lr
        self._lr_min = float(hyper["lr.min"])
        self._lr_min = g.check_limit(self._lr_min, 0, self._lr)

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        self._lr_decay_patience = int(hyper["lr.decay.patience"])
        self._lr_decay_patience = g.check_limit(
            self._lr_decay_patience, 1, self._epochs
        )

        # early stop, based on epoch
        self._early_stop_patience = int(hyper["early.stop.patience"])
        self._early_stop_patience = g.check_limit(
            self._early_stop_patience, 1, self._epochs
        )

        # number of best valid loss cnn retained
        self._keep_best_cnn_num = int(hyper["keep.best.cnn.num"])
        self._keep_best_cnn_num = g.check_limit(
            self._keep_best_cnn_num, 1, self._epochs
        )

        # augmentation percent
        self._augment_pct = float(hyper["augment.pct"])
        self._augment_pct = g.check_limit(self._augment_pct, 0, 1)

        # # empty patch percent
        # self._patch_empty_pct = float(hyper["patch.empty.pct"])
        # self._patch_empty_pct = g.check_limit(self._patch_empty_pct, 0, 1)

        # # tumor size threshold in patch
        # self._patch_tar_vol_thold = float(hyper["patch.tar.vol.thold"])
        # self._patch_tar_vol_thold = g.check_limit(self._patch_tar_vol_thold, 0, 1)

        # use cross validation or not
        self._cross_valid = bool(hyper["cross.valid"])

        # load shared hyper parameters
        super()._load_hyper(
            hyper=hyper,
            exist_cnn_path=exist_cnn_path,
        )

        # load splitting dataset, based on train/valid/test pct
        # so this must run after shared hyper loaded)
        (
            train_patients,
            valid_patients,
            test_patients,
        ) = self._load_dataset(fold, debug_mode)

        # create dataset
        train_set = BaselineDataSet(
            patient_list=train_patients,
            augment_methods=self._augment_methods,
            augment_pct=self._augment_pct,
            augment_low_limit=self._augment_low_limit,
            augment_up_limit=self._augment_up_limit,
            # patch_empty_pct=self._patch_empty_pct,
            # patch_tar_vol_thold=self._patch_tar_vol_thold,
        )
        valid_set = BaselineDataSet(patient_list=valid_patients)
        test_set = BaselineDataSet(patient_list=test_patients)

        # dataloader
        self._train_loader = DataLoader(
            dataset=train_set,
            batch_size=self._batch_size_actual,
            shuffle=True,  # only shuffle train loader
            num_workers=g.NUM_WORKERS,
        )
        self._valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=self._batch_size_actual,
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )
        self._test_loader = DataLoader(
            dataset=test_set,
            batch_size=self._batch_size_actual,
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )

    def _print_hyper(self):
        print_dict = NestedDict()
        print_dict["dropout"] = self._dropout
        print_dict["epochs"] = self._epochs
        print_dict["early.stop.patience"] = self._early_stop_patience
        print_dict["keep.best.cnn.num"] = self._keep_best_cnn_num
        print_dict["dataset.len"] = (
            self._train_loader.dataset.__len__()
            + self._valid_loader.dataset.__len__()
            + self._test_loader.dataset.__len__()
        )
        print_dict["dataset.train.len"] = self._train_loader.dataset.__len__()
        print_dict["dataset.valid.len"] = self._valid_loader.dataset.__len__()
        print_dict["dataset.test.len"] = self._test_loader.dataset.__len__()
        print_dict["dataset.k.folds"] = g.DATASET_K_FOLDS
        print_dict["cross.valid"] = self._cross_valid
        # print_dict["patch.empty.pct"] = self._patch_empty_pct
        # print_dict["patch.tar.vol.thold"] = self._patch_tar_vol_thold
        super()._print_hyper(print_dict)

    def _save_hyper(self, json_path: str):
        hyper_dict = NestedDict()
        hyper_dict["lr"] = self._lr
        hyper_dict["dataset.len"] = (
            self._train_loader.dataset.__len__()
            + self._valid_loader.dataset.__len__()
            + self._test_loader.dataset.__len__()
        )
        hyper_dict["dataset.train.len"] = self._train_loader.dataset.__len__()
        hyper_dict["dataset.valid.len"] = self._valid_loader.dataset.__len__()
        hyper_dict["dataset.test.len"] = self._test_loader.dataset.__len__()
        hyper_dict["dataset.k.folds"] = g.DATASET_K_FOLDS
        hyper_dict["cross.valid"] = self._cross_valid
        hyper_dict["epochs"] = self._epochs
        hyper_dict["epochs.actual"] = self._epochs_actual
        hyper_dict["early.stop.patience"] = self._early_stop_patience
        hyper_dict["keep.best.cnn.num"] = self._keep_best_cnn_num
        hyper_dict["dropout"] = self._dropout
        # hyper_dict["patch.empty.pct"] = self._patch_empty_pct
        # hyper_dict["patch.tar.vol.thold"] = self._patch_tar_vol_thold
        super()._save_hyper(json_path, hyper_dict)

    def loss_fig(self, baseline_id: str):
        loss_json_path = os.path.join(
            g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline", "loss.json"
        )
        self.__loss_fig(loss_json_path)

    def __loss_fig(self, loss_json_path: str):
        plt.figure().clear()

        loss_dict = g.load_json(loss_json_path)
        train_loss = []
        valid_loss = []

        for i in loss_dict:
            train_loss.append(loss_dict[i]["train"])
            valid_loss.append(loss_dict[i]["valid"])

        plt.ylim(min(train_loss) - 0.05, max(train_loss) + 0.05)
        plt.plot(range(1, len(loss_dict) + 1), train_loss, label="train")
        plt.plot(range(1, len(loss_dict) + 1), valid_loss, label="valid")
        plt.legend()
        plt.savefig(loss_json_path[:-4] + "png")

    def __training(self, cur_fold_folder: str):
        g.print_line()

        best_loss_dict = NestedDict()
        loss_save_path = os.path.join(cur_fold_folder, "loss.json")
        patience_count = 0

        for cur_epoch in range(1, self._epochs + 1):
            print("epoch: {}".format(cur_epoch))
            print("training:")
            self._cnn.train()
            sum_loss = 0
            batch_num = 0
            for inputs, labels in tqdm(self._train_loader):
                # zero grad at the begining of each mini-batch
                self._optim.zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = self._cnn(inputs)[3]
                loss = self._loss_func(outputs, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                self._optim.step()  # update param
                sum_loss += loss.item()
                batch_num += 1
            train_loss = sum_loss / batch_num

            # validation
            print("validation:")
            self._cnn.eval()
            with torch.no_grad():
                sum_loss = 0
                batch_num = 0
                for inputs, labels in tqdm(self._valid_loader):
                    inputs = inputs.to(g.DEVICE)
                    labels = labels.to(g.DEVICE)
                    outputs = self._cnn(inputs)[3]
                    loss = self._loss_func(outputs, labels)
                    sum_loss += loss.item()
                    batch_num += 1
            valid_loss = sum_loss / batch_num
            self._scheduler.step(valid_loss)

            # current epoch finished
            self._epochs_actual = cur_epoch

            # save loss in json
            loss_dict = g.load_json(loss_save_path)
            cur_epoch_loss = NestedDict()
            cur_epoch_loss["train"] = train_loss
            cur_epoch_loss["valid"] = valid_loss
            loss_dict["epoch={:03d}".format(self._epochs_actual)] = cur_epoch_loss
            g.save_json(loss_dict, loss_save_path)

            # draw loss figure
            self.__loss_fig(loss_save_path)

            # save cnn
            if len(best_loss_dict) < self._keep_best_cnn_num:
                best_loss_dict[cur_epoch] = valid_loss
                cur_epoch_folder = os.path.join(
                    cur_fold_folder, "epoch={:03d}".format(cur_epoch)
                )
                g.create_folder(cur_epoch_folder)
                self._save_cnn(os.path.join(cur_epoch_folder, "cnn.pt"))
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
                    self._save_cnn(os.path.join(cur_epoch_folder, "cnn.pt"))
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self._early_stop_patience:
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

            baseline_folder = os.path.join(
                g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline"
            )
            g.create_folder(baseline_folder)

            for cur_fold in range(1, g.DATASET_K_FOLDS + 1):
                cur_fold_folder = os.path.join(
                    baseline_folder, "fold={:02d}".format(cur_fold)
                )
                g.create_folder(cur_fold_folder)

                self._load_hyper(
                    hyper=cur_hyper,
                    fold=cur_fold,
                    debug_mode=debug_mode,
                )
                if cur_fold == 1:
                    self._print_hyper()

                g.print_line()
                print("cross validation fold: {}".format(cur_fold))

                # save an empty loss.json
                g.save_json(NestedDict(), os.path.join(cur_fold_folder, "loss.json"))

                # save hyper before training
                hyper_save_path = os.path.join(cur_fold_folder, "hyper.json")
                self._save_hyper(hyper_save_path)

                # start training
                self._time_used = datetime.now()
                self.__training(cur_fold_folder)
                self._time_used = datetime.now() - self._time_used
                self._time_used = str(self._time_used).split(".", 2)[0]

                # save hyper after training
                self._save_hyper(hyper_save_path)
                # clear time_used before next training
                self._time_used = None

                if self._cross_valid is False:
                    break

            self.inference(
                baseline_id=baseline_id,
                dataset="valid",
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
            print("current fold: ", cur_fold)
            cur_fold_folder = os.path.join(
                g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline", cur_fold_folder
            )
            cur_hyper = g.load_json(os.path.join(cur_fold_folder, "hyper.json"))

            for cur_epoch_folder in g.get_sub_folders(
                cur_fold_folder, key_word="epoch"
            ):

                cur_epoch = int(cur_epoch_folder[len("epoch=") :])
                print("current epoch: ", cur_epoch)
                cur_epoch_folder = os.path.join(cur_fold_folder, cur_epoch_folder)

                # load and print hyper
                self._load_hyper(
                    hyper=cur_hyper,
                    fold=cur_fold,
                    exist_cnn_path=os.path.join(cur_epoch_folder, "cnn.pt"),
                    debug_mode=debug_mode,
                )
                if print_hyper:
                    self._print_hyper()

                cur_score = NestedDict()
                for i in g.METRICS_LIST:
                    cur_score["avg"][i] = []

                if dataset == "test":
                    patient_list = self._test_loader.dataset.patient_list
                else:
                    patient_list = self._valid_loader.dataset.patient_list

                for cur_patient in tqdm(patient_list):
                    # if testset, create folder to save cur patient preds
                    if dataset == "test":
                        cur_patient_folder = os.path.join(
                            cur_epoch_folder,
                            "patients",
                            "patient={}".format(cur_patient),
                        )
                        g.create_folder(cur_patient_folder)

                    # result contains: "gtvs" "dsc" "msc" "hd95"
                    cur_patient_result = self._inference_single_patient(cur_patient)

                    # save score of cur patient
                    for i in g.METRICS_LIST:
                        if dataset == "test":
                            cur_score["patient={}".format(cur_patient)][
                                i
                            ] = cur_patient_result[i]
                        cur_score["avg"][i].append(cur_patient_result[i])

                    # save pred of cur patient
                    if dataset == "test":
                        for i in ["gtvs"]:  # ["gtvt", "gtvn"]:
                            g.save_nii(
                                np_data=cur_patient_result[i],
                                save_path=os.path.join(
                                    cur_patient_folder, "pred_{}.nii".format(i)
                                ),
                                spacing=g.NII_SPACING,
                            )
                            g.save_nii(
                                np_data=g.binarize_img(cur_patient_result[i]),
                                save_path=os.path.join(
                                    cur_patient_folder, "pred_{}_binary.nii".format(i)
                                ),
                                spacing=g.NII_SPACING,
                            )

                # get avg score
                for i in g.METRICS_LIST:
                    avg = cur_score["avg"][i]
                    cur_score["avg"][i] = sum(avg) / len(avg)
                # cur_score = g.load_json(
                #     os.path.join(cur_epoch_folder, "score_valid.json")
                # )

                # save score
                if dataset == "test":
                    g.save_json(
                        data=cur_score,
                        path=os.path.join(cur_epoch_folder, "score_test.json"),
                    )
                    continue

                # validation set, delete non-optimal folds and epochs
                if math.isnan(cur_score["avg"]["msd"]) or math.isnan(
                    cur_score["avg"]["hd95"]
                ):
                    g.delete_folder(cur_epoch_folder)
                    continue

                if len(best_scores) == 0:
                    save_cur_score = True
                else:
                    save_cur_score = None
                    for fold in g.get_dict_keys(best_scores):
                        for epoch in g.get_dict_keys(best_scores[fold]):
                            if (
                                cur_score["avg"]["dsc"]
                                < best_scores[fold][epoch]["dsc"]
                                and cur_score["avg"]["msd"]
                                > best_scores[fold][epoch]["msd"]
                                and cur_score["avg"]["hd95"]
                                > best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = False
                                break
                            if (
                                cur_score["avg"]["dsc"]
                                > best_scores[fold][epoch]["dsc"]
                                or cur_score["avg"]["msd"]
                                < best_scores[fold][epoch]["msd"]
                                or cur_score["avg"]["hd95"]
                                < best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = True
                            if (
                                best_scores[fold][epoch]["dsc"]
                                < cur_score["avg"]["dsc"]
                                and best_scores[fold][epoch]["msd"]
                                > cur_score["avg"]["msd"]
                                and best_scores[fold][epoch]["hd95"]
                                > cur_score["avg"]["hd95"]
                            ):
                                # delete worse score
                                g.delete_folder(
                                    os.path.join(
                                        baseline_folder,
                                        "fold={:02d}".format(fold),
                                        "epoch={:03d}".format(epoch),
                                    )
                                )
                                # best_scores[fold].pop(epoch)
                                # if len(best_scores[fold]) == 0:
                                #     best_scores.pop(fold)

                        if save_cur_score is False:
                            break

                if save_cur_score:
                    best_scores[cur_fold][cur_epoch] = cur_score["avg"]
                    g.save_json(
                        data=cur_score,
                        path=os.path.join(cur_epoch_folder, "score_valid.json"),
                    )
                else:
                    g.delete_folder(cur_epoch_folder)
