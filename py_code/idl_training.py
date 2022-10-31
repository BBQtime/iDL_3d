import os
import random
import criterion as crit
from datetime import datetime
from itertools import product
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import global_elems as g
from idl_dataset import IDLDataSet
from shared_training import SharedTraining
from nested_dict import NestedDict
from tensorboard_writer import TensorBoardWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


class IDLTraining(SharedTraining):
    def __load_next_lr(self, next_round: int):
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

    def _reset_cnn(self, cur_hyper_dict: dict, baseline_cnn_path: str):
        # RELOAD CNN
        self._cnn = super()._load_cnn(
            cnn_name=str(cur_hyper_dict["cnn.name"]),  # unet or unet++
            exist_cnn_path=baseline_cnn_path,
        )

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

    def _load_cur_hyper(
        self,
        cur_hyper_dict: dict,
        baseline_cnn_path: str,
        debug_mode: bool = False,
    ):
        if debug_mode:
            self._iter = 2  # 2 to compare difference
        else:
            self._iter = int(cur_hyper_dict["iter"])
            g.check_limit(self._iter, 1, None)

        # min lr
        self._lr_min = float(cur_hyper_dict["lr.min"])

        # reset lr before next round or not
        self._lr_reset = bool(cur_hyper_dict["lr.reset"])

        # lr list
        self._lr = cur_hyper_dict["lr"]
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
        self._lr_decay_patience = int(cur_hyper_dict["lr.decay.patience"])
        g.check_limit(self._lr_decay_patience, 1, self._iter)

        # augmentation times (based on after shared hyper)
        if debug_mode:
            self._augment_times = 2
        else:
            self._augment_times = int(cur_hyper_dict["augment.times"])
            self._augment_times = g.check_limit(self._augment_times, 1, None)

        # augmentation percent (based on augment_times)
        self._augment_pct = self._augment_times / (self._augment_times + 1)

        # freeze layers
        self._layer_freezing = bool(cur_hyper_dict["layer.freezing"])

        # post processing
        self._post_process = bool(cur_hyper_dict["post.process"])

        # select step
        if debug_mode:
            self._select_step = [2, 1]
        else:
            self._select_step = cur_hyper_dict["select.step"]
            # select.step is saved in json file as a string, not a list, because:
            # (1) it's easier to read the json file (only one line)
            # (2) a "list" will be recognized as multiple hyper parameters,
            # then start multiple training
            self._select_step = g.str_to_list(self._select_step)
            for i in range(len(self._select_step)):
                self._select_step[i] = int(self._select_step[i])
                self._select_step[i] = g.check_limit(self._select_step[i], 1, None)

        # select scenario
        self._select_scenario = str(cur_hyper_dict["select.scenario"]).lower()
        if (
            self._select_scenario != "largest"
            and self._select_scenario != "equal.divide"
        ):
            self._select_scenario = "random"

        # load shared hyper
        super()._load_cur_hyper(
            cur_hyper_dict=cur_hyper_dict,
            exist_cnn_path=baseline_cnn_path,
        )

        # # split dataset, based on train/valid/test pct
        # must after shared hyper loaded)
        self._patient_list = self._load_dataset(debug_mode)[2]

    def _print_hyper(self):
        print_dict = NestedDict()
        print_dict["idl id:"] = self._idl_id
        print_dict["num of patients:"] = len(self._patient_list)
        print_dict["iter:"] = self._iter
        print_dict["augment times:"] = self._augment_times
        print_dict["slice select step:"] = self._select_step
        print_dict["slice select scenario:"] = self._select_scenario
        print_dict["layer freezing:"] = self._layer_freezing
        print_dict["reset lr:"] = self._lr_reset
        print_dict["post processing:"] = self._post_process
        print_dict["dropout baseline:"] = self._dropout_baseline
        print_dict["dropout idl:"] = self._dropout
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
        hyper_dict["post.process"] = self._post_process
        hyper_dict["dropout.baseline"] = self._dropout_baseline
        hyper_dict["dropout.idl"] = self._dropout
        super()._save_hyper(json_path, hyper_dict)

    # create a folder contains imgs in IDL_RESULTS_FOLDER
    def baseline_visualize(
        self,
        baseline_id: str,
        idl_results_folder: str,
        idl_id: str,
    ):
        g.print_line()
        print("baseline visualization")
        print("baseline training id: " + baseline_id)
        # get baseline cnn and hyper path
        cnn_path, hyper_path = self.__get_baseline_paths(baseline_id)
        hyper_dict = g.load_json(hyper_path)

        # load cnn and batch size
        self._cnn = self._load_cnn(
            cnn_name=str(hyper_dict["cnn.name"]),  # unet or unet++
            exist_cnn_path=cnn_path,
        )
        self._batch_size, self._batch_size_actual = self._load_batch_size(hyper_dict)

        # create result folder
        if idl_results_folder is None or idl_results_folder == "":
            idl_results_folder = g.IDL_RESULTS_FOLDER
        cur_result_folder = os.path.join(idl_results_folder, idl_id)
        g.create_folder(cur_result_folder)

        patient_list = self._load_dataset(debug_mode=False)
        patient_list = patient_list[2]  # test set only
        for cur_patient in tqdm(patient_list):
            # save result of round=0 and iter=0
            self.__test_process(
                cur_result_folder=cur_result_folder,
                cur_patient=cur_patient,
                cur_round=0,
                cur_iter=0,
                save_img=True,
            )

    def real_training(
        self,
        baseline_id: str,
        idl_results_folder: str,
        idl_id: str,
        cur_patient: str,
        cur_round: int,
        debug_mode: bool = False,
    ):
        self._idl_id = idl_id

        # get baseline cnn and hyper path
        baseline_cnn_path, baseline_hyper_path = self.__get_baseline_paths(baseline_id)
        g.print_line()
        print(baseline_cnn_path)
        # load hypers
        idl_hyper_dict = g.load_json(g.IDL_HYPER_JSON)
        baseline_hyper_dict = g.load_json(baseline_hyper_path)

        # add important baseline hypers to iDL hypers
        for i in ["cnn.name"]:
            idl_hyper_dict[i] = baseline_hyper_dict[i]

        # make sure all hypers are unique, no arrangement
        cur_hyper_dict = NestedDict()
        for i in idl_hyper_dict:
            if isinstance(idl_hyper_dict[i], list):
                cur_hyper_dict[i] = idl_hyper_dict[i][0]
            else:
                cur_hyper_dict[i] = idl_hyper_dict[i]

        # load and print hyper
        self._load_cur_hyper(
            cur_hyper_dict=cur_hyper_dict,
            baseline_cnn_path=baseline_cnn_path,
            debug_mode=debug_mode,
        )
        self._print_hyper()

        # check if result folder exist
        cur_result_folder = os.path.join(idl_results_folder, self._idl_id)
        if not os.path.exists(cur_result_folder):
            g.exit_app("IDLTraining.real_training(): iDL result folder doesn't exist")

        # create json file to save train loss
        train_loss_dict = NestedDict()
        train_loss_dict["iter"] = NestedDict()
        g.save_json(
            train_loss_dict,
            os.path.join(
                cur_result_folder, "patient={}".format(cur_patient), "train_loss.json"
            ),
        )

        # get annotated slices
        cur_round_folder = os.path.join(
            cur_result_folder,
            "patient={}".format(cur_patient),
            "round={:02d}".format(cur_round),
        )
        annotated_slices = NestedDict()
        annotated_slices["round=01"] = []  # doesn't matter what the dict key is
        for file_name in g.get_sub_files(cur_round_folder, key_word="_label.npy"):
            slice_id = file_name[len("slice_") : -len("_label.npy")]
            slice_id = slice_id.zfill(3)
            annotated_slices["round=01"].append(slice_id)

        # training start time
        self._time_used = datetime.now()

        self.__train_process_cur_round(
            cur_result_folder=cur_result_folder,
            cur_patient=cur_patient,
            annotated_slices=annotated_slices,
            label_folder=cur_round_folder,
        )

        # get training time used before save hyper
        self._time_used = datetime.now() - self._time_used
        # save hyper
        self._save_hyper(os.path.join(cur_result_folder, "hyper.json"))

    def simulation(
        self,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        group_start_time = self._init_start_time()

        # load iDL hyper
        idl_full_hyper_dict = self._load_full_hyper(g.IDL_HYPER_JSON)

        # get hyper_keys to combine with hyper_values to create "cur_hyper_combination" later
        idl_hyper_keys = g.get_dict_keys(idl_full_hyper_dict)

        # get all Cartesian Product of hyper dict values
        for cur_hyper_values in product(*idl_full_hyper_dict.values()):

            # create current hyper param dict
            cur_hyper_dict = NestedDict()
            for i in range(len(cur_hyper_values)):
                cur_hyper_dict[idl_hyper_keys[i]] = cur_hyper_values[i]

            # get baseline cnn and hyper path
            baseline_id = cur_hyper_dict["baseline.id"]
            baseline_cnn_path, baseline_hyper_path = self.__get_baseline_paths(
                baseline_id
            )
            g.print_line()
            print(baseline_cnn_path)

            # load baseline hyper
            baseline_hyper_dict = g.load_json(baseline_hyper_path)

            # add important baseline hypers to cur iDL hypers
            cur_hyper_dict["cnn.name"] = baseline_hyper_dict["cnn.name"]

            # record baseline dropout rate
            self._dropout_baseline = baseline_hyper_dict["dropout"]

            # idl_id must be generated before "_print_hyper()"
            self._idl_id = self._init_train_id(
                group_start_time=group_start_time,
                train_remark=train_remark,
                debug_mode=debug_mode,
                full_hyper_dict=idl_full_hyper_dict,
                cur_hyper_dict=cur_hyper_dict,
            )

            # load and print hyper
            self._load_cur_hyper(
                cur_hyper_dict=cur_hyper_dict,
                baseline_cnn_path=baseline_cnn_path,
                debug_mode=debug_mode,
            )
            g.print_line()
            self._print_hyper()

            # create result folder
            cur_result_folder = os.path.join(g.IDL_RESULTS_FOLDER, self._idl_id)
            g.create_folder(cur_result_folder)

            # training start time
            self._time_used = datetime.now()

            # patient loop
            for cur_patient in self._patient_list:
                self.__train_process_cur_patient(
                    cur_patient=cur_patient,
                    cur_result_folder=cur_result_folder,
                )

                # reload all hyper (including cnn/optimizer/scheduler)
                # before next patient (except the last patient)
                idx = self._patient_list.index(cur_patient)
                if (idx + 1) < len(self._patient_list):
                    self._reset_cnn(
                        cur_hyper_dict=cur_hyper_dict,
                        baseline_cnn_path=baseline_cnn_path,
                    )

            # get training time used before save hyper
            self._time_used = datetime.now() - self._time_used
            # save hyper in result folder
            self._save_hyper(os.path.join(cur_result_folder, "hyper.json"))

            # write tensorboard and boxplt after iDL is done
            # self.tensorboard_record_loss(self._idl_id)
            # self.tensorboard_record_score(self._idl_id)
            self.json_record_avg_score(self._idl_id)

    def __get_baseline_paths(self, baseline_id: str):
        # get baseline cnn and hyper path
        baseline_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id)
        file_list = g.get_sub_files(baseline_folder)
        for file_name in file_list:
            if file_name.endswith(".pt"):
                baseline_cnn_path = os.path.join(baseline_folder, file_name)
            if file_name == "hyper.json":
                baseline_hyper_path = os.path.join(baseline_folder, file_name)
        return baseline_cnn_path, baseline_hyper_path

    def __get_cur_round_slices(
        self, patient: str, cur_result_folder: str, annotated_slices: dict
    ):
        # here in this function,
        # slices of cur round have not been added into annotated_slices
        cur_round_slices = []
        cur_round = len(annotated_slices) + 1

        # get available slices and 2d dsc
        cur_patient_folder = os.path.join(
            cur_result_folder,
            "patient={}".format(patient),
        )
        if cur_round == 1:
            prev_round_folder = os.path.join(cur_patient_folder, "baseline")
        else:
            prev_round_folder = os.path.join(
                cur_patient_folder,
                "round={:02d}".format(cur_round - 1),
            )

        # get max iteration
        if cur_round > 1:
            max_iter = len(g.get_sub_files(prev_round_folder, key_word=".json"))
        else:
            max_iter = 0
        prev_score_json_path = os.path.join(
            prev_round_folder,
            "iter={:02d}.json".format(max_iter),
        )
        candidate_slices = g.load_json(prev_score_json_path)["dsc"]["2d"]

        # remove slices already annotated
        for i in annotated_slices:
            for slice_id in annotated_slices[i]:
                if slice_id in candidate_slices:
                    candidate_slices.pop(slice_id)

        # put tumor size of each slice into "tumor_size_dict"
        # also remove slices without tumor
        tumor_size_dict = NestedDict()
        for cur_slice in candidate_slices.copy():
            label_path = os.path.join(g.DATASET_FOLDER, patient, cur_slice, "label.npy")
            cur_tumor_size = np.load(label_path).astype(int).sum()
            if cur_tumor_size > 0:
                tumor_size_dict[cur_slice] = cur_tumor_size

            # remove empty slices
            if candidate_slices[cur_slice] == "empty":
                candidate_slices.pop(cur_slice)
            # if "no.pred" or "no.label", 2d dsc of cur_slice = 0
            elif (
                candidate_slices[cur_slice] == "no.pred"
                or candidate_slices[cur_slice] == "no.label"
            ):
                candidate_slices[cur_slice] = 0

        # choose equal divide slices (only for the first select step)
        if self._select_scenario == "equal.divide" and cur_round == 1:
            divided_parts = self._select_step[0] + 1
            candidate_slices = g.get_dict_keys(candidate_slices)
            for i in range(divided_parts - 1):
                idx = len(candidate_slices) * (i + 1) / divided_parts
                idx = round(idx)
                idx = g.check_limit(idx, 1, len(candidate_slices))
                cur_round_slices.append(candidate_slices[idx - 1])

        # random select order
        elif self._select_scenario == "random":
            candidate_slices = g.get_dict_keys(candidate_slices)
            random.shuffle(candidate_slices)
            for i in candidate_slices:
                # get the first elements of each tuple (list of slice id)
                if i not in cur_round_slices:
                    cur_round_slices.append(i)

        # (1) "largest"
        # (2) "equal.divide", cur_round >= 2
        else:
            # descrease sort the dict (return a list of tuple)
            tumor_size_desc_tuple = sorted(
                tumor_size_dict.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
            for i in tumor_size_desc_tuple:
                # get the first elements of each tuple (list of slice id)
                if i[0] not in cur_round_slices:
                    cur_round_slices.append(i[0])

        # narrow cur_round_slices based on select.step
        if self._select_step[cur_round - 1] < len(cur_round_slices):
            cur_round_slices = cur_round_slices[: self._select_step[cur_round - 1]]

        return cur_round_slices

    def __train_process_cur_patient(
        self,
        cur_patient: str,
        cur_result_folder: str,
    ):
        # create current patient folder
        cur_patient_folder = os.path.join(
            cur_result_folder, "patient={}".format(cur_patient)
        )
        g.create_folder(cur_patient_folder)
        g.print_line()
        print("patient:", cur_patient)

        # create json file to save train loss
        train_loss_dict = NestedDict()
        train_loss_dict["iter"] = NestedDict()
        g.save_json(
            train_loss_dict, os.path.join(cur_patient_folder, "train_loss.json")
        )

        # save baseline score and imgs
        # g.print_line()
        self.__test_process(
            cur_result_folder=cur_result_folder,
            cur_patient=cur_patient,
            cur_round=0,
            cur_iter=0,
            save_img=True,
        )

        if self._post_process:
            # filter fp <= 2 slices and save pred
            filtered_slices, post_process_folder = self.__filter_fp(
                cur_result_folder=cur_result_folder,
                cur_patient=cur_patient,
            )
            # post processing only when find fp slices
            if len(filtered_slices["post.process"]) > 0:
                print("post processing:")
                self.__train_process_cur_round(
                    cur_result_folder=cur_result_folder,
                    cur_patient=cur_patient,
                    annotated_slices=filtered_slices,
                    label_folder=post_process_folder,
                )
                self.__load_next_lr(next_round=1)

        # start iDL
        annotated_slices = NestedDict()
        # loop through each round
        for i in range(len(self._select_step)):
            cur_round_slices = self.__get_cur_round_slices(
                patient=cur_patient,
                cur_result_folder=cur_result_folder,
                annotated_slices=annotated_slices,
            )
            if len(cur_round_slices) == 0:
                break

            # add cur round slices into annotated_slices(Dict)
            cur_round = "round={:02d}".format(i + 1)
            annotated_slices[cur_round] = cur_round_slices

            print("round:", i + 1)
            self.__train_process_cur_round(
                cur_result_folder=cur_result_folder,
                cur_patient=cur_patient,
                annotated_slices=annotated_slices,
                label_folder=g.DATASET_FOLDER,
            )

            # load new lr before next round
            if self._lr_reset:
                self.__load_next_lr(next_round=i + 2)

        # save annotated slices in cur patient folder
        # change list into string
        for i in annotated_slices:
            annotated_slices[i] = g.list_to_str(annotated_slices[i])
        # save path
        json_save_path = os.path.join(
            cur_result_folder, "patient={}".format(cur_patient), "annotated_slices.json"
        )
        g.save_json(data=annotated_slices, path=json_save_path)

    def __filter_fp(
        self,
        cur_result_folder: str,
        cur_patient: str,
    ):
        cur_patient_folder = os.path.join(
            cur_result_folder, "patient={}".format(cur_patient)
        )
        origin_pred = g.load_nii(
            os.path.join(
                cur_patient_folder,
                "baseline",
                "pred.nii",
            )
        )
        # print(origin_pred.min(), origin_pred.max())

        filtered_pred = np.zeros_like(origin_pred)

        filtered_slices = NestedDict()
        filtered_slices["post.process"] = []

        all_ccs = g.get_connected_components(origin_pred)
        for cur_cc in all_ccs:
            cur_cc_slices = []
            for i in range(cur_cc.shape[0]):
                if cur_cc[i].sum() > 0:
                    cur_cc_slices.append(i)

            if len(cur_cc_slices) > 1:
                # filtered_pred += cur_cc
                filtered_pred = np.logical_or(filtered_pred, cur_cc).astype(int)
            else:
                filtered_slices["post.process"] += cur_cc_slices
        # print(filtered_pred.min(), filtered_pred.max())
        # g.show_img(filtered_pred)
        # g.save_nii(origin_pred, "F:/origin_pred.nii", g.NII_SPACING)
        # g.save_nii(filtered_pred, "F:/filtered_pred.nii", g.NII_SPACING)

        if len(filtered_slices["post.process"]) > 0:
            # create filter.fp folder
            filter_fp_folder = os.path.join(cur_patient_folder, "filter.fp")
            g.create_folder(filter_fp_folder)
            # save filtered prediction
            g.save_nii(
                np_data=filtered_pred,
                save_path=os.path.join(filter_fp_folder, "pred.nii"),
                spacing=g.NII_SPACING,
            )

            # create post processing folder
            post_process_folder = os.path.join(cur_patient_folder, "post.process")
            g.create_folder(post_process_folder)
            # save filtered slices
            for i in filtered_slices["post.process"]:
                # g.show_img(origin_pred[i], print_info=True)
                # g.show_img(filtered_pred[i], print_info=True)
                np.save(
                    os.path.join(
                        post_process_folder, "slice_{:02d}_label.npy".format(i)
                    ),
                    filtered_pred[i],
                )

            # change slices id from int to str
            for i in range(len(filtered_slices["post.process"])):
                filtered_slices["post.process"][i] = "{:03d}".format(
                    filtered_slices["post.process"][i]
                )
        else:
            post_process_folder = None

        return filtered_slices, post_process_folder

    def __train_process_cur_round(
        self,
        cur_result_folder: str,
        cur_patient: str,
        annotated_slices: dict,
        label_folder: str,
    ):
        # len(annotated_slices) will always > 0
        if len(annotated_slices) == 1:
            if g.get_dict_keys(annotated_slices)[0] == "post.process":
                cur_round = -1
            else:
                cur_round = 1
        else:
            cur_round = len(annotated_slices)

        # create iDL dataset
        idl_dataset = IDLDataSet(
            patient=cur_patient,
            slice_dict=annotated_slices,
            label_folder=label_folder,
            augment_times=self._augment_times,
            augment_pct=self._augment_pct,
            augment_method=self._augment_method,
            augment_low_limit=self._augment_low_limit,
            augment_up_limit=self._augment_up_limit,
        )

        # optimize batch size (before create dataloader)
        optim_batch_size = self._optimize_batch_size(idl_dataset)

        # idlive dataloader
        idl_loader = DataLoader(
            dataset=idl_dataset,
            batch_size=optim_batch_size,
            shuffle=True,
            num_workers=g.NUM_WORKERS,
        )

        # iter loop
        for cur_iter in tqdm(range(self._iter)):
            iter_time_used = datetime.now()
            sum_loss = 0
            batch_num = 0

            # switch to train mode before training
            self._cnn.train()

            # freeze layers before iDL
            if self._layer_freezing:
                # here, self._cnn is DataParallel, not network itself
                if g.used_gpu_count() > 1:
                    self._cnn.module.freeze_top()
                else:
                    self._cnn.freeze_top()

            for inputs, labels in idl_loader:
                # zero grad at the begining of each mini-batch
                self._optim.zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = self._cnn(inputs)
                # if 1:
                #     concat_img = torch.cat(
                #         [
                #             inputs[0, 0, :, :],  # ct
                #             inputs[0, 1, :, :],  # pt
                #             labels[0, 0, :, :],  # label
                #             outputs[0, 0, :, :],  # pred
                #         ],
                #         dim=1,
                #     )
                #     g.show_img(concat_img, print_info=False)
                loss = self._loss_func(outputs, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                self._optim.step()  # update param
                sum_loss += loss.item()
                batch_num += 1

            # cur iteration finished
            train_loss = sum_loss / batch_num
            self._scheduler.step(train_loss)
            iter_time_used = datetime.now() - iter_time_used

            # save loss data in json file
            loss_save_path = os.path.join(
                cur_result_folder, "patient={}".format(cur_patient), "train_loss.json"
            )
            train_loss_dict = g.load_json(loss_save_path)

            # normal iDL without post processing
            if cur_round >= 1:
                train_loss_dict["iter"][
                    "{:03d}".format((cur_round - 1) * self._iter + (cur_iter + 1))
                ] = train_loss
            # post processing, cur_round=-1
            else:
                train_loss_dict["iter"]["{:03d}".format(cur_iter + 1)] = train_loss
            g.save_json(train_loss_dict, loss_save_path)

            # test cnn and save scores of current iter
            if cur_iter + 1 == self._iter:
                save_img = True
            else:
                save_img = False
            self.__test_process(
                cur_result_folder=cur_result_folder,
                cur_patient=cur_patient,
                cur_round=cur_round,
                cur_iter=cur_iter + 1,
                save_img=save_img,
                iter_time_used=iter_time_used,
            )

    def __test_process(
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

    def tensorboard_record_loss(self, idl_id: str):
        g.print_line()
        print("write loss into tensorboard")
        result_folder = os.path.join(g.IDL_RESULTS_FOLDER, idl_id)

        avg_loss_dict = NestedDict()
        patient_folder_list = g.get_sub_folders(result_folder)
        for cur_patient_folder in tqdm(patient_folder_list):
            cur_loss_dict = g.load_json(
                os.path.join(result_folder, cur_patient_folder, "train_loss.json")
            )["iter"]

            # record loss of create cur patient
            cur_writer = TensorBoardWriter(
                os.path.join(g.IDL_TENSORBOARD_FOLDER, cur_patient_folder, "loss")
            )
            for i in cur_loss_dict:
                cur_writer.write_loss_per_iter(
                    idl_id=idl_id, train_loss=cur_loss_dict[i], _iter=int(i)
                )

            # add cur_loss into avg_loss
            if len(avg_loss_dict) == 0:
                for i in cur_loss_dict:
                    avg_loss_dict[i] = []
                    avg_loss_dict[i].append(cur_loss_dict[i])
            else:
                for i in cur_loss_dict:
                    avg_loss_dict[i].append(cur_loss_dict[i])

        avg_writer = TensorBoardWriter(
            os.path.join(g.IDL_TENSORBOARD_FOLDER, "avg", "loss")
        )
        for i in g.get_dict_keys(avg_loss_dict):
            avg_loss_dict[i] = g.get_avg_value(avg_loss_dict[i])
            avg_writer.write_loss_per_iter(
                idl_id=idl_id, train_loss=avg_loss_dict[i], _iter=int(i)
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
                for score_type in ["dsc", "msd", "hd95"]:
                    avg_score_dict[score_type][cur_round]["excluding.annotated"][
                        cur_patient
                    ] = crit.test_3d_score(
                        cur_round_pred,
                        cur_round_label,
                        score_type=score_type,
                        binarize=True,
                    )

                    avg_score_dict[score_type][cur_round]["full.slices"][
                        cur_patient
                    ] = iter_json_data[score_type]["3d"]

                    for i in ["excluding.annotated", "full.slices"]:
                        if (
                            avg_score_dict[score_type][cur_round][i][cur_patient]
                            == "no.pred"
                            or avg_score_dict[score_type][cur_round][i][cur_patient]
                            == "no.label"
                        ):
                            if score_type == "dsc":
                                avg_score_dict[score_type][cur_round][i][
                                    cur_patient
                                ] = 0.0
                            else:
                                avg_score_dict[score_type][cur_round][i][
                                    cur_patient
                                ] = g.IMG_SIZE

                        elif (
                            avg_score_dict[score_type][cur_round][i][cur_patient]
                            == "empty"
                        ):
                            if score_type == "dsc":
                                avg_score_dict[score_type][cur_round][i][
                                    cur_patient
                                ] = 1.0
                            else:
                                avg_score_dict[score_type][cur_round][i][
                                    cur_patient
                                ] = 0.0

        # save score of each patient
        g.save_json(
            avg_score_dict, os.path.join(train_result_folder, "score_per_patient.json")
        )

        # calculate avg value after all patients data recorded
        for score_type in ["dsc", "msd", "hd95"]:
            for slice_type in ["excluding.annotated", "full.slices"]:
                for cur_round in avg_score_dict[score_type]:
                    cur_round_avg_score = avg_score_dict[score_type][cur_round][
                        slice_type
                    ].values()
                    cur_round_avg_score = sum(cur_round_avg_score) / len(
                        cur_round_avg_score
                    )
                    avg_score_dict[score_type][cur_round][
                        slice_type
                    ] = cur_round_avg_score

        g.save_json(avg_score_dict, os.path.join(train_result_folder, "avg_score.json"))

    def tensorboard_record_score(self, idl_id: str):
        g.print_line()
        print("write score into tensorboard")
        train_result_folder = os.path.join(g.IDL_RESULTS_FOLDER, idl_id)

        # read training iteration from json
        hyper_path = os.path.join(train_result_folder, "hyper.json")
        hyper_dict = g.load_json(hyper_path)
        base_iter = hyper_dict["iter"]

        # save avg 2d/3d dsc/msd/hd95
        avg_score_dict = NestedDict()

        # loop through patients
        patient_folder_list = g.get_sub_folders(train_result_folder)

        for cur_patient_folder in tqdm(patient_folder_list):

            # get patient id
            cur_patient = cur_patient_folder[cur_patient_folder.find("=") + 1 :]

            # tensorboard writer of current patient
            cur_patient_writer = NestedDict()
            for dim in ["3d", "2d"]:
                cur_patient_writer[dim] = TensorBoardWriter(
                    os.path.join(
                        g.IDL_TENSORBOARD_FOLDER,
                        cur_patient_folder,
                        "{}.score".format(dim),
                    )
                )

            # check round folder list
            round_folder_list = g.get_sub_folders(
                os.path.join(train_result_folder, cur_patient_folder)
            )
            annotated_slices = g.load_json(
                os.path.join(
                    train_result_folder, cur_patient_folder, "annotated_slices.json"
                )
            )

            # round = len(annotated_slices) = len(round_folder_list) - 1
            # len(round_folder_list) - 1, because there is a "baseline" folder
            if len(annotated_slices) != (len(round_folder_list) - 1):
                g.exit_app(
                    "update round folders may be missing, idl_id={}, patient={}".format(
                        idl_id, cur_patient
                    )
                )

            # record the sum iterations of former rounds
            former_iter_sum = 0

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
                cur_round = round_folder_list.index(cur_round_folder)
                err_str = "idl_id={}, patient={}, {}".format(
                    idl_id, cur_patient, cur_round_folder
                )
                err_str = "iter json file may be missing, " + err_str

                if cur_round > 0:
                    cur_round_iter = len(iter_json_list)
                    # check if any iter json file is missing
                    # "cur_round_iter" should be an integer multiple of "base_iter"
                    if cur_round_iter % base_iter != 0:
                        g.exit_app(err_str)

                # if cur_round=0, there is only 1 iter json file
                else:
                    cur_round_iter = 0
                    # check if any iter json file is missing
                    if len(iter_json_list) != 1:
                        g.exit_app(err_str)

                # loop through iter json files
                for cur_iter_json in iter_json_list:
                    cur_iter = int(cur_iter_json[len("iter=") : -len(".json")])
                    json_data = g.load_json(
                        os.path.join(
                            train_result_folder,
                            cur_patient_folder,
                            cur_round_folder,
                            cur_iter_json,
                        )
                    )
                    self.__tensorboard_record_cur_iter(
                        idl_id=idl_id,
                        cur_patient=cur_patient,
                        tensorboard_writer=cur_patient_writer,
                        cur_round=cur_round,
                        former_iter_sum=former_iter_sum,
                        cur_round_iter=cur_round_iter,
                        cur_iter=cur_iter,
                        json_data=json_data,
                        avg_score_dict=avg_score_dict,
                    )

                # current round finished, add up iteration
                former_iter_sum += cur_round_iter

        # record avg 2d/3d results
        for dim in ["2d", "3d"]:
            avg_writer = TensorBoardWriter(
                os.path.join(g.IDL_TENSORBOARD_FOLDER, "avg", "{}.score".format(dim))
            )

            for score_type in ["dsc", "msd", "hd95"]:
                former_iter_sum = 0

                # loop through rounds
                for cur_round in avg_score_dict[dim][score_type]:
                    if cur_round > 0:
                        cur_round_iter = len(avg_score_dict[dim][score_type][cur_round])
                    else:
                        cur_round_iter = 0

                    # loop through iterations
                    for cur_iter in avg_score_dict[dim][score_type][cur_round]:
                        avg_value = avg_score_dict[dim][score_type][cur_round][
                            cur_iter
                        ].values()
                        avg_value = sum(avg_value) / len(avg_value)

                        # record score round mapping
                        if cur_round == 0 or cur_iter == cur_round_iter:
                            avg_writer.write_score_per_round(
                                idl_id=idl_id,
                                score_type=score_type,
                                value=avg_value,
                                round=cur_round,
                            )
                        # write score iter mapping
                        avg_writer.write_score_per_iter(
                            idl_id=idl_id,
                            score_type=score_type,
                            value=avg_value,
                            former_iter_sum=former_iter_sum,
                            cur_iter=cur_iter,
                        )
                    # cur round finished, accumulate iters
                    former_iter_sum += cur_round_iter

    def __tensorboard_record_cur_iter(
        self,
        idl_id: str,
        cur_patient: str,
        tensorboard_writer: TensorBoardWriter,
        cur_round: int,
        former_iter_sum: int,
        cur_round_iter: int,
        cur_iter: int,
        json_data: dict,
        avg_score_dict: dict,
    ):
        for score_type in ["dsc", "msd", "hd95"]:

            for i in ["2d.avg", "3d"]:
                if (
                    json_data[score_type][i] == "no.pred"
                    or json_data[score_type][i] == "no.label"
                ):
                    if score_type == "dsc":
                        json_data[score_type][i] = 0.0
                    else:
                        json_data[score_type][i] = g.IMG_SIZE
                elif json_data[score_type][i] == "empty":
                    if score_type == "dsc":
                        json_data[score_type][i] = 1.0
                    else:
                        json_data[score_type][i] = 0.0

            # record result for avg average calculation
            avg_score_dict["3d"][score_type][cur_round][cur_iter][
                cur_patient
            ] = json_data[score_type]["3d"]
            avg_score_dict["2d"][score_type][cur_round][cur_iter][
                cur_patient
            ] = json_data[score_type]["2d.avg"]

            # (cur_round == 0) means: baseline
            # (cur_iter == total_iter) means: cur round finished
            if cur_round == 0 or cur_iter == cur_round_iter:
                # 3d score round mapping
                tensorboard_writer["3d"].write_score_per_round(
                    idl_id=idl_id,
                    score_type=score_type,
                    value=json_data[score_type]["3d"],
                    round=cur_round,
                )
                # 2d score round mapping
                tensorboard_writer["2d"].write_score_per_round(
                    idl_id=idl_id,
                    score_type=score_type,
                    value=json_data[score_type]["2d.avg"],
                    round=cur_round,
                )

            # 3d score iter mapping
            tensorboard_writer["3d"].write_score_per_iter(
                idl_id=idl_id,
                score_type=score_type,
                value=json_data[score_type]["3d"],
                former_iter_sum=former_iter_sum,
                cur_iter=cur_iter,
            )
            # 2d score iter mapping
            tensorboard_writer["2d"].write_score_per_iter(
                idl_id=idl_id,
                score_type=score_type,
                value=json_data[score_type]["2d.avg"],
                former_iter_sum=former_iter_sum,
                cur_iter=cur_iter,
            )
