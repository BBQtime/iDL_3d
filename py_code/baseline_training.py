import global_elems as g
import os
import torch
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from itertools import product
from nested_dict import NestedDict
from torch.utils.data import DataLoader
from shared_training import SharedTraining
from tensorboard_writer import TensorBoardWriter
from baseline_dataset import BaselineDataSet


class BaselineTraining(SharedTraining):
    def __init__(self):
        super().__init__()

        # record self._epochs_actual because of "early.stop.patience"
        self._epochs_actual = 0
        self.__tensorboard_writer = TensorBoardWriter(g.BASELINE_TENSORBOARD_FOLDER)

    def _load_cur_hyper(
        self,
        cur_hyper_dict: dict,
        exist_cnn_path: str = None,
        debug_mode: bool = False,
    ):

        # epochs
        if debug_mode:
            self._epochs = 2  # run 2 epochs to compare difference in loss
        else:
            self._epochs = int(cur_hyper_dict["epochs"])
            self._epochs = g.check_limit(self._epochs, 1, None)

        # lr
        self._lr = float(cur_hyper_dict["lr"])
        self._lr = g.check_limit(self._lr, 1e-10, None)
        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._lr_actual = self._lr * used_gpu_count
        else:
            self._lr_actual = self._lr

        # min lr
        self._lr_min = float(cur_hyper_dict["lr.min"])
        self._lr_min = g.check_limit(self._lr_min, 0.0, self._lr)

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        self._lr_decay_patience = int(cur_hyper_dict["lr.decay.patience"])
        self._lr_decay_patience = g.check_limit(
            self._lr_decay_patience, 1, self._epochs
        )

        # early stop, based on epoch
        self._early_stop_patience = int(cur_hyper_dict["early.stop.patience"])
        self._early_stop_patience = g.check_limit(
            self._early_stop_patience, 1, self._epochs
        )

        # augmentation percent
        self._augment_pct = float(cur_hyper_dict["augment.pct"])
        self._augment_pct = g.check_limit(self._augment_pct, 0.0, 1.0)

        # load shared hyper parameters
        super()._load_cur_hyper(
            cur_hyper_dict=cur_hyper_dict,
            exist_cnn_path=exist_cnn_path,
        )

        # load splitting dataset, based on train/valid/test pct
        # so this must run after shared hyper loaded)
        (
            train_patient_list,
            valid_patient_list,
            test_patient_list,
        ) = self._load_dataset(debug_mode)

        # create dataset
        train_set = BaselineDataSet(
            patient_list=train_patient_list,
            augment_pct=self._augment_pct,
            augment_method=self._augment_method,
            augment_low_limit=self._augment_low_limit,
            augment_up_limit=self._augment_up_limit,
        )
        valid_set = BaselineDataSet(patient_list=valid_patient_list)
        test_set = BaselineDataSet(patient_list=test_patient_list)

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
        print_dict["baseline id:"] = self._baseline_id
        print_dict["dropout:"] = self._dropout
        print_dict["epochs:"] = self._epochs
        print_dict["early stopping patience:"] = self._early_stop_patience
        print_dict["dataset len:"] = (
            self._train_loader.dataset.__len__()
            + self._valid_loader.dataset.__len__()
            + self._test_loader.dataset.__len__()
        )
        print_dict["train set len:"] = self._train_loader.dataset.__len__()
        print_dict["valid set len:"] = self._valid_loader.dataset.__len__()
        print_dict["test set len:"] = self._test_loader.dataset.__len__()
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
        hyper_dict["epochs"] = self._epochs
        hyper_dict["epochs.actual"] = self._epochs_actual
        hyper_dict["early.stop.patience"] = self._early_stop_patience
        hyper_dict["dropout"] = self._dropout
        super()._save_hyper(json_path, hyper_dict)

    def __create_result_folder(self, baseline_id: str):
        result_path = os.path.join(g.BASELINE_RESULTS_FOLDER, baseline_id)
        g.create_folder(result_path)
        cnn_save_path = os.path.join(result_path, "epoch=")
        hyper_save_path = os.path.join(result_path, "hyper.json")
        loss_save_path = os.path.join(result_path, "train_loss.json")
        # create json file
        train_loss_dict = NestedDict()
        train_loss_dict["epoch"] = NestedDict()
        g.save_json(train_loss_dict, loss_save_path)
        return cnn_save_path, hyper_save_path, loss_save_path

    def __train_process(self, cnn_save_path: str, loss_save_path: str):
        g.clear_gpu_cache()
        g.print_line()

        self._time_used = datetime.now()
        best_epoch = None
        best_loss = None
        patience_count = 0

        for cur_epoch in range(1, self._epochs + 1):
            print("epoch={}".format(cur_epoch))
            print("training:")
            self._cnn.train()
            sum_loss = 0
            batch_num = 0
            for inputs, labels in tqdm(self._train_loader):
                # zero grad at the begining of each mini-batch
                self._optim.zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = self._cnn(inputs)
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
                    outputs = self._cnn(inputs)
                    loss = self._loss_func(outputs, labels)
                    sum_loss += loss.item()
                    batch_num += 1
            valid_loss = sum_loss / batch_num
            self._scheduler.step(valid_loss)

            # current epoch finished
            self._epochs_actual = cur_epoch

            # write tensorboard (loss/epoch map)
            self.__tensorboard_writer.write_loss_per_epoch(
                baseline_id=self._baseline_id,
                train_loss=train_loss,
                valid_loss=valid_loss,
                epoch=self._epochs_actual,
            )

            # save loss data in json file
            train_loss_dict = g.load_json(loss_save_path)
            cur_loss_dict = NestedDict()
            cur_loss_dict["train.loss"] = train_loss
            cur_loss_dict["valid.loss"] = valid_loss
            train_loss_dict["epoch"][
                "{:03d}".format(self._epochs_actual)
            ] = cur_loss_dict
            g.save_json(train_loss_dict, loss_save_path)

            # save cnn
            if best_loss is None:
                best_loss = valid_loss
                best_epoch = cur_epoch
                self._save_cnn(cnn_save_path + str(best_epoch) + ".pt")
            else:
                if valid_loss < best_loss:
                    os.remove(cnn_save_path + str(best_epoch) + ".pt")
                    best_loss = valid_loss
                    best_epoch = cur_epoch
                    self._save_cnn(cnn_save_path + str(best_epoch) + ".pt")
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self._early_stop_patience:
                        break

        # training over
        self._time_used = datetime.now() - self._time_used
        g.clear_gpu_cache()

    def train(
        self,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        baseline_id_list = []
        group_start_time = self._init_start_time()

        # load hyper dict
        full_hyper_dict = self._load_full_hyper(g.BASELINE_HYPER_JSON)
        # get hyper_keys to combine with hyper_values to create "cur_hyper_combination" later
        hyper_keys = g.get_dict_keys(full_hyper_dict)

        # get all cartesian products of hyper dict values
        for cur_hyper_values in product(*full_hyper_dict.values()):

            # create current hyper param combination
            cur_hyper_dict = NestedDict()
            for i in range(len(cur_hyper_values)):
                cur_hyper_dict[hyper_keys[i]] = cur_hyper_values[i]

            # baseline_id must be generated before "_print_hyper()"
            self._baseline_id = self._init_train_id(
                group_start_time=group_start_time,
                train_remark=train_remark,
                debug_mode=debug_mode,
                full_hyper_dict=full_hyper_dict,
                cur_hyper_dict=cur_hyper_dict,
            )
            baseline_id_list.append(self._baseline_id)

            self._load_cur_hyper(
                cur_hyper_dict=cur_hyper_dict,
                debug_mode=debug_mode,
            )
            g.print_line()
            self._print_hyper()

            # create result folder, generate cnn and hyper save path
            (
                cnn_save_path,
                hyper_save_path,
                loss_save_path,
            ) = self.__create_result_folder(self._baseline_id)

            # start current training
            self.__train_process(cnn_save_path, loss_save_path)

            self._save_hyper(hyper_save_path)

            self.test(
                baseline_id=self._baseline_id,
                print_hyper=False,
                debug_mode=debug_mode,
            )

        return baseline_id_list

    # test dsc/msd/hd95
    def test(
        self,
        baseline_id: str,
        print_hyper: bool = True,
        debug_mode: bool = False,
    ):
        self._baseline_id = baseline_id
        cur_train_folder = os.path.join(g.BASELINE_RESULTS_FOLDER, self._baseline_id)

        hyper_path = os.path.join(cur_train_folder, "hyper.json")
        cur_hyper_dict = g.load_json(hyper_path)

        # init results
        best_cnn_path = None
        cur_scores = NestedDict()
        best_scores = NestedDict()
        for dim in ["2d", "3d"]:
            best_scores["dsc"][dim] = 0.0
            best_scores["msd"][dim] = g.IMG_SIZE
            best_scores["hd95"][dim] = g.IMG_SIZE

        # test each cnn
        file_list = g.get_sub_files(cur_train_folder)
        for file_name in file_list:
            if not file_name.endswith(".pt"):
                continue
            cur_cnn_path = os.path.join(cur_train_folder, file_name)
            g.print_line()
            print(cur_cnn_path)

            # load and print hyper
            self._load_cur_hyper(
                cur_hyper_dict=cur_hyper_dict,
                exist_cnn_path=cur_cnn_path,
                debug_mode=debug_mode,
            )
            if print_hyper:
                self._print_hyper()

            # get score of each patient, also save imgs
            all_patient_results = self._test_patients(
                patient_list=self._test_loader.dataset.patient_list,
                cnn=self._cnn,
                imgs_save_folder=None,
                show_tqdm_bar=True,
            )

            # clear cur results before calculate score
            for score_type in ["dsc", "msd", "hd95"]:
                for dim in ["2d", "3d"]:
                    cur_scores[score_type][dim] = []

            # calculate average (2d.avg and 3d) score of all patients
            for score_type in ["dsc", "msd", "hd95"]:
                for patient in all_patient_results:
                    cur_scores[score_type]["2d"].append(
                        all_patient_results[patient][score_type]["2d.avg"]
                    )
                    cur_scores[score_type]["3d"].append(
                        all_patient_results[patient][score_type]["3d"]
                    )

                for dim in ["2d", "3d"]:
                    sum_score = 0
                    score_count = 0
                    for cur_score in cur_scores[score_type][dim]:
                        if g.is_number(cur_score):
                            sum_score += cur_score
                            score_count += 1
                        else:
                            score_count += 1

                    if score_count == 0:
                        cur_scores[score_type][dim] = "empty"
                    else:
                        cur_scores[score_type][dim] = sum_score / score_count

                    if g.is_number(cur_scores[score_type][dim]):
                        if score_type == "dsc":
                            print(
                                "{} {}: {:.2f}".format(
                                    dim, score_type, cur_scores[score_type][dim]
                                )
                            )
                        else:
                            print(
                                "{} {}: {:.1f}".format(
                                    dim, score_type, cur_scores[score_type][dim]
                                )
                            )
                    else:
                        print("{} {}: empty".format(dim, score_type))

            # record best scores
            if (
                best_cnn_path is None
                or cur_scores["dsc"]["3d"] > best_scores["dsc"]["3d"]
                or (
                    cur_scores["dsc"]["3d"] == best_scores["dsc"]["3d"]
                    and cur_scores["dsc"]["2d"] > best_scores["dsc"]["2d"]
                )
            ):
                # remove the second best cnn
                if best_cnn_path is not None:
                    os.remove(best_cnn_path)

                # update the best cnn path and scores
                best_cnn_path = cur_cnn_path
                for score_type in ["dsc", "msd", "hd95"]:
                    for dim in ["2d", "3d"]:
                        best_scores[score_type][dim] = cur_scores[score_type][dim]

            # not the best cnn, remove
            else:
                os.remove(cur_cnn_path)

        best_epoch = best_cnn_path[
            best_cnn_path.find("epoch=") + len("epoch=") : -len(".pt")
        ]
        best_scores["best.epoch"] = best_epoch

        g.save_json(
            data=best_scores, path=os.path.join(cur_train_folder, "test_score.json")
        )
