import os
import random
import torch
from datetime import datetime
from itertools import product
import numpy as np
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import global_elems as g
from idl_dataset import IDLDataSet
from shared_training import SharedTraining
from nested_dict import NestedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau


class IDLTraining(SharedTraining):
    def __load_next_lr(self, next_round: int):

        # self._lr is a list of lr of each round
        if next_round > len(self._lr):
            next_round = len(self._lr)

        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._lr_actual = self._lr[next_round - 1] * used_gpu_count
        else:
            self._lr_actual = self._lr[next_round - 1]

        # OPTIMIZER (no need to move to cuda)
        self._optim = optim.Adam(params=self._cnn.parameters(), lr=self._lr_actual)

        # SCHEDULER
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        self._scheduler = ReduceLROnPlateau(
            optimizer=self._optim,
            mode="min",
            factor=self._lr_decay_factor,  # "factor=1.0" will cause an error
            patience=self._lr_decay_patience,
            min_lr=self._lr_min,
        )

    def _reset_cnn(self, hyper: dict, baseline_cnn_path: str):
        # RELOAD CNN
        self._cnn = super()._load_cnn(baseline_cnn_path)

        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._lr_actual = self._lr[0] * used_gpu_count
        else:
            self._lr_actual = self._lr[0]

        # optimizer (no need to move to cuda)
        self._optim = optim.Adam(params=self._cnn.parameters(), lr=self._lr_actual)

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        self._scheduler = ReduceLROnPlateau(
            optimizer=self._optim,
            mode="min",
            factor=self._lr_decay_factor,  # "factor=1.0" will cause an error
            patience=self._lr_decay_patience,
            min_lr=self._lr_min,
        )

    def _load_hyper(
        self,
        hyper: dict,
        baseline_cnn_path: str,
        debug_mode: bool = False,
    ):
        if debug_mode:
            self._iter = 2  # 2 to compare difference
        else:
            self._iter = int(hyper["iter"])
            g.check_limit(self._iter, 1, None)

        # min lr
        self._lr_min = float(hyper["lr.min"])

        # reset lr before next round or not
        self._lr_reset = bool(hyper["lr.reset"])

        # lr list
        self._lr = hyper["lr"]
        # lr list is saved in json file as a string, not a list, because:
        # (1) string IS easier to read the json file (only one line)
        # (2) a "list" will be recognized as multiple hyper parameters
        self._lr = g.str_to_list(self._lr)
        for i in range(len(self._lr)):
            self._lr[i] = float(self._lr[i])
            self._lr[i] = g.check_limit(self._lr[i], 1e-10, None)
            # check lr_min, make sure it is lower than any lr in lr_step
            self._lr_min = g.check_limit(self._lr_min, 0.0, self._lr[i])

        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._lr_actual = self._lr[0] * used_gpu_count
        else:
            self._lr_actual = self._lr[0]

        # lr decay patience (before shared hyper)
        self._lr_decay_patience = int(hyper["lr.decay.patience"])
        g.check_limit(self._lr_decay_patience, 1, self._iter)

        # augmentation times (based on after shared hyper)
        if debug_mode:
            self._augment_times = 2
        else:
            self._augment_times = int(hyper["augment.times"])
            self._augment_times = g.check_limit(self._augment_times, 1, None)

        # augmentation percent (based on augment_times)
        self._augment_pct = self._augment_times / (self._augment_times + 1)

        # freeze layers
        self._layer_freezing = bool(hyper["layer.freezing"])

        # select step
        if debug_mode:
            self._select_step = [2, 1]
        else:
            self._select_step = hyper["select.step"]
            # select.step is saved in json file as a string, not a list, because:
            # (1) it's easier to read the json file (only one line)
            # (2) a "list" will be recognized as multiple hyper parameters,
            # then start multiple training
            self._select_step = g.str_to_list(self._select_step)
            for i in range(len(self._select_step)):
                self._select_step[i] = int(self._select_step[i])
                self._select_step[i] = g.check_limit(self._select_step[i], 1, None)

        # select scenario
        self._select_scenario = str(hyper["select.scenario"]).lower()
        if (
            self._select_scenario != "largest"
            and self._select_scenario != "equal.divide"
        ):
            self._select_scenario = "random"

        # load shared hyper
        super()._load_hyper(
            hyper=hyper,
            exist_cnn_path=baseline_cnn_path,
        )

        # # split dataset, based on train/valid/test pct
        # must after shared hyper loaded)
        self._patient_list = self._load_dataset(debug_mode)[2]

    def _print_hyper(self):
        print_dict = NestedDict()
        print_dict["num of patients:"] = len(self._patient_list)
        print_dict["iter:"] = self._iter
        print_dict["augment times:"] = self._augment_times
        print_dict["slice select step:"] = self._select_step
        print_dict["slice select scenario:"] = self._select_scenario
        print_dict["layer freezing:"] = self._layer_freezing
        print_dict["reset lr:"] = self._lr_reset
        print_dict["dropout:"] = self._dropout
        super()._print_hyper(print_dict)

    def _save_hyper(self, json_path: str):
        hyper_dict = NestedDict()
        hyper_dict["augment.times"] = self._augment_times
        hyper_dict["iter"] = self._iter
        hyper_dict["lr"] = g.list_to_str(self._lr)
        hyper_dict["lr.reset"] = self._lr_reset
        hyper_dict["select.step"] = g.list_to_str(self._select_step)
        hyper_dict["select.scenario"] = self._select_scenario
        hyper_dict["layer.freezing"] = self._layer_freezing
        hyper_dict["dropout"] = self._dropout
        super()._save_hyper(json_path, hyper_dict)

    # def real_training(
    #     self,
    #     baseline_id: str,
    #     idl_results_folder: str,
    #     idl_id: str,
    #     cur_patient: str,
    #     cur_round: int,
    #     debug_mode: bool = False,
    # ):
    #     self._idl_id = idl_id

    #     # get baseline cnn and hyper path
    #     baseline_cnn_path, baseline_hyper_path = self.__get_baseline_paths(baseline_id)
    #     g.print_line()
    #     print(baseline_cnn_path)
    #     # load hypers
    #     idl_hyper_dict = g.load_json(g.IDL_HYPER_JSON)
    #     baseline_hyper_dict = g.load_json(baseline_hyper_path)

    #     # make sure all hypers are unique, no arrangement
    #     hyper = NestedDict()
    #     for i in idl_hyper_dict:
    #         if isinstance(idl_hyper_dict[i], list):
    #             hyper[i] = idl_hyper_dict[i][0]
    #         else:
    #             hyper[i] = idl_hyper_dict[i]

    #     # load and print hyper
    #     self._load_hyper(
    #         hyper=hyper,
    #         baseline_cnn_path=baseline_cnn_path,
    #         debug_mode=debug_mode,
    #     )
    #     self._print_hyper()

    #     # check if result folder exist
    #     cur_result_folder = os.path.join(idl_results_folder, self._idl_id)
    #     if not os.path.exists(cur_result_folder):
    #         g.exit_app("IDLTraining.real_training(): iDL result folder doesn't exist")

    #     # create json file to save train loss
    #     train_loss_dict = NestedDict()
    #     train_loss_dict["iter"] = NestedDict()
    #     g.save_json(
    #         train_loss_dict,
    #         os.path.join(
    #             cur_result_folder, "patient={}".format(cur_patient), "train_loss.json"
    #         ),
    #     )

    #     # get annotated slices
    #     cur_round_folder = os.path.join(
    #         cur_result_folder,
    #         "patient={}".format(cur_patient),
    #         "round={:02d}".format(cur_round),
    #     )
    #     annotated_slices = NestedDict()
    #     annotated_slices["round=01"] = []  # doesn't matter what the dict key is
    #     for file_name in g.get_sub_files(cur_round_folder, key_word="_label.npy"):
    #         slice_id = file_name[len("slice_") : -len("_label.npy")]
    #         slice_id = slice_id.zfill(3)
    #         annotated_slices["round=01"].append(slice_id)

    #     # training start time
    #     self._time_used = datetime.now()

    #     self.__training_cur_round(
    #         cur_result_folder=cur_result_folder,
    #         cur_patient=cur_patient,
    #         annotated_slices=annotated_slices,
    #         label_folder=cur_round_folder,
    #     )

    #     # get training time used before save hyper
    #     self._time_used = datetime.now() - self._time_used
    #     # save hyper
    #     self._save_hyper(os.path.join(cur_result_folder, "hyper.json"))

    # return a list of string
    # here in this function,
    # slices of cur round have not been added into annotated_slices
    def __select_cur_round_slices(
        self, patient_folder: str, annotated_slices: dict
    ) -> list:

        cur_round_slices = []
        candidate_slices = NestedDict()  # {"slice_id": pred_tumor_size}
        cur_round = len(annotated_slices) + 1

        patient = Path(patient_folder).name
        patient = patient[len("patient=") :]

        # get available slices and 2d dsc
        if cur_round == 1:
            pred_folder = Path(patient_folder).parent.parent.parent
            pred_folder = os.path.join(
                pred_folder, "baseline", "patients", "patient={}".format(patient)
            )
        else:
            pred_folder = os.path.join(patient_folder, "round={:02d}".format(cur_round))

        pred = g.load_nii(os.path.join(pred_folder, "pred_gtvs.nii"), binary=True)
        label = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient))
        )

        # go through pred and record tumor size
        annotated_slices = g.dict_to_list(annotated_slices)
        for cur_slice in range(pred.shape[0]):
            # skip slice that already been annotated
            if cur_slice in annotated_slices:
                continue
            else:
                pred_slice_tumor_size = pred[cur_slice].sum()
                if pred_slice_tumor_size > 0:
                    candidate_slices[cur_slice] = pred_slice_tumor_size
                elif label[cur_slice].sum() > 0:
                    candidate_slices[cur_slice] = 0

        # "equal.divide", round = 1
        if self._select_scenario == "equal.divide" and cur_round == 1:
            divided_parts = self._select_step[0] + 1
            candidate_slices = g.get_dict_keys(candidate_slices)
            for i in range(1, divided_parts):
                idx = len(candidate_slices) * i / divided_parts
                idx = round(idx)
                idx = g.check_limit(idx, 1, len(candidate_slices))
                cur_round_slices.append(candidate_slices[idx - 1])

        # "random"
        elif self._select_scenario == "random":
            cur_round_slices = g.get_dict_keys(candidate_slices)
            random.shuffle(cur_round_slices)

        # (1) "largest"
        # (2) "equal.divide", round >= 2
        else:
            # descrease sort the dict (return a list of tuple)
            candidate_slices = g.sort_dict_by_value(candidate_slices, reverse=True)
            cur_round_slices = g.get_dict_keys(candidate_slices)

        # narrow cur_round_slices based on select.step
        cur_round_slices_num = self._select_step[cur_round - 1]
        if cur_round_slices_num < len(cur_round_slices):
            cur_round_slices = cur_round_slices[:cur_round_slices_num]

        return cur_round_slices

    def __training_cur_round(
        self,
        cur_round_folder: str,
        annotated_slices: dict,
        label_folder: str,
    ):
        cur_round = Path(cur_round_folder).name
        cur_round = int(cur_round[len("round=") :])

        patient = Path(cur_round_folder).parent.name
        patient = patient[len("patient=") :]

        loss_json_path = os.path.join(Path(cur_round_folder).parent, "loss.json")
        loss_dict = g.load_json(loss_json_path)

        if cur_round == 1:
            pred_folder = Path(cur_round_folder).parent.parent.parent.parent
            pred_folder = os.path.join(
                pred_folder, "baseline", "patients", "patient={}".format(patient)
            )
        else:
            pred_folder = os.path.join(
                Path(cur_round_folder).parent, "round={:02d}".format(cur_round - 1)
            )

        cur_round_time_used = datetime.now()

        # create iDL dataset
        idl_dataset = IDLDataSet(
            patient=patient,
            annotated_slices=annotated_slices,
            label_folder=label_folder,
            pred_folder=pred_folder,
            ignore_other_anotated_slices=False,
            augment_methods=self._augment_methods,
            augment_times=self._augment_times,
            augment_pct=self._augment_pct,
            augment_low_limit=self._augment_low_limit,
            augment_up_limit=self._augment_up_limit,
        )

        # optimize batch size (before create dataloader)
        self._batch_size_actual = self._optimize_batch_size(idl_dataset)

        # idl dataloader
        idl_loader = DataLoader(
            dataset=idl_dataset,
            batch_size=self._batch_size_actual,
            shuffle=True,
            num_workers=g.NUM_WORKERS,
        )

        # iter loop
        for cur_iter in tqdm(range(self._iter)):
            self._cnn.train()
            sum_loss = 0
            batch_num = 0

            # freeze layers before iDL
            if self._layer_freezing:
                if g.used_gpu_count() > 1:
                    # here, self._cnn is DataParallel, not network itself
                    self._cnn.module.freeze_top()
                else:
                    self._cnn.freeze_top()

            for inputs, labels, reserved_slices in idl_loader:
                # zero grad at the begining of each mini-batch
                self._optim.zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = self._cnn(inputs)

                mask = torch.ones_like(labels)
                # remove non-annotated slices in outputs
                for b in range(labels.shape[0]):  # batch
                    for c in range(labels.shape[1]):  # channel
                        for d in range(labels.shape[2]):  # depth
                            if labels[b][c][d].sum() == 0:
                                mask[b][c][d] = mask[b][c][d] - mask[b][c][d]
                                # outputs[b][c][d] += torch.finfo(torch.float32).eps
                            else:
                                print("123")

                outputs = outputs * mask
                loss = self._loss_func(outputs, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                self._optim.step()  # update param
                sum_loss += loss.item()
                batch_num += 1

            # cur iter finished
            # update scheduler
            iter_loss = sum_loss / batch_num
            self._scheduler.step(iter_loss)
            # record loss
            loss_dict[
                "iter={:03d}".format((cur_round - 1) * self._iter + (cur_iter + 1))
            ] = iter_loss

        # current round finished
        # inference
        self.inference(
            cur_result_folder=cur_result_folder,
            cur_patient=cur_patient,
            cur_round=cur_round,
            cur_iter=cur_iter + 1,
            save_img=save_img,
            iter_time_used=iter_time_used,
        )
        # save time used
        cur_round_time_used = datetime.now() - cur_round_time_used
        cur_round_time_used = str(cur_round_time_used).split(".", 2)[0]
        time_save_path = os.path.join(Path(cur_round_folder).parent, "time_used.json")
        time_used_dict = g.load_json(time_save_path)
        time_used_dict["round={:02d}".format(cur_round)] = cur_round_time_used
        g.save_json(time_used_dict, time_save_path)

        # save loss
        g.save_json(loss_dict, loss_json_path)

    def __training_cur_patient(
        self,
        patient: str,
        idl_folder: str,
    ):
        # create current patient folder
        patient_folder = os.path.join(
            idl_folder, "patients", "patient={}".format(patient)
        )
        g.create_folder(patient_folder)
        # create a json to save time used for cur patient
        g.save_json(NestedDict(), os.path.join(patient_folder, "time_used.json"))
        # create an empty loss.json
        g.save_json(NestedDict(), os.path.join(patient_folder, "loss.json"))

        g.print_line()
        print("patient:", patient)

        annotated_slices = NestedDict()

        # loop through each round
        for cur_round in range(1, len(self._select_step) + 1):

            cur_round_slices = self.__select_cur_round_slices(
                patient_folder=patient_folder,
                annotated_slices=annotated_slices,
            )
            if len(cur_round_slices) == 0:
                break

            # start current round
            print("round:", cur_round + 1)

            # add cur_round_slices into annotated_slices BEFORE __training_cur_round()
            annotated_slices["round={:02d}".format(cur_round)] = cur_round_slices

            self.__training_cur_round(
                cur_round_folder=os.path.join(
                    patient_folder, "round={:02d}".format(cur_round)
                ),
                annotated_slices=annotated_slices,
                label_folder=g.DATASET_FOLDER,
            )

            # load new lr before next round
            if self._lr_reset:
                self.__load_next_lr(next_round=cur_round + 1)

        # save annotated slices in cur patient folder
        for i in annotated_slices:
            annotated_slices[i] = g.list_to_str(annotated_slices[i])
        g.save_json(
            data=annotated_slices,
            path=os.path.join(
                idl_folder, "patient={}".format(patient), "annotated_slices.json"
            ),
        )

    def simulation(
        self,
        baseline_id: str,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        for hyper in self._load_group_hyper(g.IDL_HYPER_JSON):

            baseline_cnn_path = g.get_sub_files(
                os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline"),
                key_word=".pt",
                return_full_path=True,
            )[0]

            self._load_hyper(
                hyper=hyper,
                baseline_cnn_path=baseline_cnn_path,
                debug_mode=debug_mode,
            )
            g.print_line()
            self._print_hyper()

            idl_id = "idl_" + self._init_train_id(
                train_remark=train_remark,
                debug_mode=debug_mode,
                hyper_json_path=g.IDL_HYPER_JSON,
                hyper=hyper,
            )

            # create idl result folder
            idl_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, idl_id)
            g.create_folder(idl_folder)

            # save hyper before training
            hyper_save_path = os.path.join(idl_folder, "hyper.json")
            self._save_hyper(hyper_save_path)

            # training start time
            self._time_used = datetime.now()

            # patient loop
            for patient in self._patient_list:

                self.__training_cur_patient(
                    patient=patient,
                    idl_folder=idl_folder,
                )

                # reset cnn/optimizer/scheduler before next patient
                if patient != self._patient_list[-1]:
                    self._reset_cnn(
                        hyper=hyper,
                        baseline_cnn_path=baseline_cnn_path,
                    )

            # get training time used before save hyper
            self._time_used = datetime.now() - self._time_used
            self._time_used = str(self._time_used).split(".", 2)[0]
            self._save_hyper(hyper_save_path)
            # self.json_record_avg_score(self._idl_id)

    def inference(
        self,
        cur_result_folder: str,
        cur_patient: str,
        cur_round: int,
        cur_iter: int,
        save_img: bool = False,
        iter_time_used: datetime = None,
    ):
        cur_patient_folder = os.path.join(
            cur_result_folder, "patient={}".format(cur_patient)
        )
        if cur_round == 0:
            cur_round_folder = os.path.join(
                cur_patient_folder,
                "baseline",
            )
        elif cur_round == -1:
            cur_round_folder = os.path.join(
                cur_patient_folder,
                "post.process",
            )
        else:
            cur_round_folder = os.path.join(
                cur_patient_folder,
                "round={:02d}".format(cur_round),
            )
        if not cur_round_folder.endswith("/"):
            cur_round_folder += "/"
        g.create_folder(cur_round_folder)

        # get test score and also save imgs
        if save_img:
            imgs_save_folder = cur_round_folder
        else:
            imgs_save_folder = None

        # baseline, save all imgs
        if cur_round == 0:
            save_pred_only = False
        # post processing or iDL, save pred only
        else:
            save_pred_only = True

        result_dict = self._inference_single_patient(
            patient_list=[cur_patient],
            cnn=self._cnn,
            imgs_save_folder=imgs_save_folder,
            save_pred_only=save_pred_only,
            show_tqdm_bar=False,
        )[cur_patient]

        # time used
        if iter_time_used is not None:
            result_dict["iter.time.used"] = g.keep_decimal(iter_time_used, 2)
        # save test result
        g.save_json(
            data=result_dict,
            path=cur_round_folder + "iter={:02d}.json".format(cur_iter),
        )

    # this funciton ignores post processing,
    # only calculate avg value of round=0/1/2/3/....
    def json_record_avg_score(self, idl_id: str):
        g.print_line()
        print("record avg score into json")

        # save avg 3d score of dsc/msd/hd95
        avg_score_dict = NestedDict()

        train_result_folder = os.path.join(g.IDL_RESULTS_FOLDER, idl_id)

        # get iteration from hyper param json
        hyper_path = os.path.join(train_result_folder, "hyper.json")
        hyper_dict = g.load_json(hyper_path)
        base_iter = hyper_dict["iter"]

        patient_folder_list = g.get_sub_folders(train_result_folder)

        # loop through patients
        for cur_patient_folder in tqdm(patient_folder_list):

            cur_patient = cur_patient_folder[cur_patient_folder.find("=") + 1 :]

            round_folder_list = ["baseline"]
            round_folder_list += g.get_sub_folders(
                os.path.join(train_result_folder, cur_patient_folder),
                key_word="round=",
            )

            # load label.nii of cur patient
            cur_patient_label = g.load_nii(
                os.path.join(
                    train_result_folder,
                    cur_patient_folder,
                    "baseline",
                    "label.nii",
                )
            )

            cur_patient_annotated_slices = g.load_json(
                os.path.join(
                    train_result_folder, cur_patient_folder, "annotated_slices.json"
                )
            )

            # round = len(annotated_slices) = len(round_folder_list)-1
            if len(cur_patient_annotated_slices) != len(round_folder_list) - 1:
                g.exit_app(
                    "round folders may be missing, idl_id={}, patient={}".format(
                        idl_id, cur_patient
                    )
                )

            # remove annotated slices from pred and label
            for i in cur_patient_annotated_slices:
                cur_patient_annotated_slices[i] = g.str_to_list(
                    cur_patient_annotated_slices[i]
                )
            cur_patient_annotated_slices = g.dict_to_list(cur_patient_annotated_slices)
            cur_patient_annotated_slices.sort(reverse=True)

            # loop through round folders
            for cur_round_folder in round_folder_list:
                iter_json_list = g.get_sub_files(
                    os.path.join(
                        train_result_folder,
                        cur_patient_folder,
                        cur_round_folder,
                    ),
                    key_word=".json",
                )

                # get iter of current round
                err_str = "idl_id={}, patient={}, {}".format(
                    idl_id, cur_patient, cur_round_folder
                )
                err_str = "iter json file may be missing, " + err_str

                if cur_round_folder == "baseline":
                    cur_round = 0
                else:
                    cur_round = int(cur_round_folder[-2:])

                if cur_round > 0:
                    cur_round_iter = len(iter_json_list)
                    # check if any iter json file is missing
                    # "cur_round_iter" should be an integer multiple of "base_iter"
                    if cur_round_iter % base_iter != 0:
                        g.exit_app(err_str)

                else:  # if cur_round=0, there is only 1 iter json file
                    # check if any iter json file is missing
                    if len(iter_json_list) != 1:
                        g.exit_app(err_str)

                cur_round = "round={:02}".format(cur_round)

                iter_json_data = g.load_json(
                    os.path.join(
                        train_result_folder,
                        cur_patient_folder,
                        cur_round_folder,
                        iter_json_list[-1],
                    )
                )

                # pred.nii path of cur round
                cur_round_pred = g.load_nii(
                    os.path.join(
                        train_result_folder,
                        cur_patient_folder,
                        cur_round_folder,
                        "pred.nii",
                    )
                )
                cur_round_label = cur_patient_label

                # remove annotated slices from label and pred
                for i in cur_patient_annotated_slices:
                    # print(cur_patient_final_pred.shape)
                    cur_round_label = np.delete(cur_round_label, obj=int(i), axis=0)
                    cur_round_pred = np.delete(cur_round_pred, obj=int(i), axis=0)
                    # print(cur_patient_final_pred.shape)

                # calculate scores excluding annotated slices
                for metric_type in g.METRICS_LIST:
                    avg_score_dict[metric_type][cur_round]["excluding.annotated"][
                        cur_patient
                    ] = crit.test_3d_score(
                        cur_round_pred,
                        cur_round_label,
                        metric_type=metric_type,
                        binarize=True,
                    )

                    avg_score_dict[metric_type][cur_round]["full.slices"][
                        cur_patient
                    ] = iter_json_data[metric_type]["3d"]

                    for i in ["excluding.annotated", "full.slices"]:
                        if (
                            avg_score_dict[metric_type][cur_round][i][cur_patient]
                            == "no.pred"
                            or avg_score_dict[metric_type][cur_round][i][cur_patient]
                            == "no.label"
                        ):
                            if metric_type == "dsc":
                                avg_score_dict[metric_type][cur_round][i][
                                    cur_patient
                                ] = 0.0
                            else:
                                avg_score_dict[metric_type][cur_round][i][
                                    cur_patient
                                ] = g.IMG_SIZE

                        elif (
                            avg_score_dict[metric_type][cur_round][i][cur_patient]
                            == "empty"
                        ):
                            if metric_type == "dsc":
                                avg_score_dict[metric_type][cur_round][i][
                                    cur_patient
                                ] = 1.0
                            else:
                                avg_score_dict[metric_type][cur_round][i][
                                    cur_patient
                                ] = 0.0

        # save score of each patient
        g.save_json(
            avg_score_dict, os.path.join(train_result_folder, "score_per_patient.json")
        )

        # calculate avg value after all patients data recorded
        for metric_type in g.METRICS_LIST:
            for slice_type in ["excluding.annotated", "full.slices"]:
                for cur_round in avg_score_dict[metric_type]:
                    cur_round_avg_score = avg_score_dict[metric_type][cur_round][
                        slice_type
                    ].values()
                    cur_round_avg_score = sum(cur_round_avg_score) / len(
                        cur_round_avg_score
                    )
                    avg_score_dict[metric_type][cur_round][
                        slice_type
                    ] = cur_round_avg_score

        g.save_json(avg_score_dict, os.path.join(train_result_folder, "avg_score.json"))


# simulated iDL
if 1:
    idl = IDLTraining()
    idl.simulation(
        baseline_id="baseline_2022.11.27.06.23.46_target.vol.pct=0_lr=0.0005",
        train_remark="delete.this",
        debug_mode=1,
    )
