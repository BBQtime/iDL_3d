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
    def __load_next_round_lr(self, next_round: int, hyper: NestedDict):
        # hyper["lr"] is a list of lr of each round
        if next_round > len(hyper["lr"]):
            next_round = len(hyper["lr"])

        if g.used_gpu_count() > 1:
            hyper["lr.actual"] = hyper["lr"][next_round - 1] * g.used_gpu_count()
        else:
            hyper["lr.actual"] = hyper["lr"][next_round - 1]

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr.actual"]
        )

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        hyper["scheduler"] = ReduceLROnPlateau(
            optimizer=hyper["optim"],
            mode="min",
            factor=hyper["lr.decay.factor"],  # "factor=1" will cause an error
            patience=hyper["lr.decay.patience"],
            min_lr=hyper["lr.min"],
        )

    def __reset_cnn(self, hyper: dict, baseline_cnn_path: str):
        # RELOAD CNN
        super()._load_cnn(hyper, baseline_cnn_path)

        if g.used_gpu_count() > 1:
            hyper["lr.actual"] = hyper["lr"][0] * g.used_gpu_count()
        else:
            hyper["lr.actual"] = hyper["lr"][0]

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr.actual"]
        )

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        hyper["scheduler"] = ReduceLROnPlateau(
            optimizer=hyper["optim"],
            mode="min",
            factor=hyper["lr.decay.factor"],  # "factor=1" will cause an error
            patience=hyper["lr.decay.patience"],
            min_lr=hyper["lr.min"],
        )

    def __load_hyper(
        self, hyper: dict, baseline_cnn_path: str, debug_mode: bool = False
    ):
        if debug_mode:
            # iter = 2 to compare difference
            hyper["iter"] = 2
        else:
            hyper["iter"] = int(hyper["iter"])
            g.check_limit(hyper["iter"], 1, None)

        # min lr
        hyper["lr.min"] = float(hyper["lr.min"])

        # reset lr before next round or not
        hyper["lr.reset"] = bool(hyper["lr.reset"])

        # lr (list)
        # the list of lr is saved in json file as a string, not a list, because:
        # (1) string is easier to read the json file (only one line)
        # (2) a "list" will be recognized as multiple hyper parameters
        hyper["lr"] = g.str_to_list(hyper["lr"])
        for i in range(len(hyper["lr"])):
            hyper["lr"][i] = float(hyper["lr"][i])
            hyper["lr"][i] = g.check_limit(hyper["lr"][i], 1e-10, None)
            # check min lr, make sure it is lower than any lr in the lr list
            hyper["lr.min"] = g.check_limit(hyper["lr.min"], 0, hyper["lr"][i])

        if g.used_gpu_count() > 1:
            hyper["lr.actual"] = hyper["lr"][0] * g.used_gpu_count()
        else:
            hyper["lr.actual"] = hyper["lr"][0]

        # lr decay patience (before shared hyper)
        hyper["lr.decay.patience"] = int(hyper["lr.decay.patience"])
        g.check_limit(hyper["lr.decay.patience"], 1, hyper["iter"])

        # augmentation times (based on after shared hyper)
        if debug_mode:
            hyper["augment.times"] = 2
        else:
            hyper["augment.times"] = int(hyper["augment.times"])
            hyper["augment.times"] = g.check_limit(hyper["augment.times"], 1, None)

        # augmentation percent (based on augment_times)
        hyper["augment.pct"] = hyper["augment.times"] / (hyper["augment.times"] + 1)

        # freeze layers
        hyper["layer.freezing"] = bool(hyper["layer.freezing"])

        # select step
        if debug_mode:
            hyper["select.step"] = [2, 1]
        else:
            hyper["select.step"] = hyper["select.step"]
            # select.step is saved in json file as a string, not a list, because:
            # (1) it's easier to read the json file (only one line)
            # (2) a "list" will be recognized as multiple hyper parameters,
            # then start multiple training
            hyper["select.step"] = g.str_to_list(hyper["select.step"])
            for i in range(len(hyper["select.step"])):
                hyper["select.step"][i] = int(hyper["select.step"][i])
                hyper["select.step"][i] = g.check_limit(
                    hyper["select.step"][i], 1, None
                )

        # select scenario
        hyper["select.scenario"] = str(hyper["select.scenario"]).lower()
        if (
            hyper["select.scenario"] != "largest"
            and hyper["select.scenario"] != "equal.divide"
        ):
            hyper["select.scenario"] = "random"

        # load shared hyper
        super()._load_hyper(
            hyper=hyper,
            exist_cnn_path=baseline_cnn_path,
        )

        # run this after shared hyper loaded, actual batch size is needed
        hyper["patients"] = self.__load_dataset(hyper=hyper, debug_mode=debug_mode)

    def __load_dataset(self, hyper: NestedDict, debug_mode: bool = False):
        json_data = g.load_json(g.DATASET_SPLITTING_JSON)
        test_patients = g.str_to_list(json_data["test.patients"])

        if debug_mode:
            test_patients = test_patients[: hyper["batch.size.actual"]]

        return test_patients

    def __get_simple_hyper(self, hyper: NestedDict) -> NestedDict:
        simple_hyper = NestedDict()
        for i in hyper:
            if i == "lr":
                simple_hyper[i] = g.list_to_str(hyper[i])
            elif i == "patients":
                simple_hyper[i] = len(hyper[i])
            else:
                simple_hyper[i] = hyper[i]
        return simple_hyper

    def __print_hyper(self, hyper: NestedDict):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def __save_hyper(self, hyper: NestedDict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._save_hyper(simple_hyper, json_path)

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
    #     idl_hyper_dict = g.load_json(g.HYPER_JSON_IDL)
    #     baseline_hyper_dict = g.load_json(baseline_hyper_path)

    #     # make sure all hypers are unique, no arrangement
    #     hyper = NestedDict()
    #     for i in idl_hyper_dict:
    #         if isinstance(idl_hyper_dict[i], list):
    #             hyper[i] = idl_hyper_dict[i][0]
    #         else:
    #             hyper[i] = idl_hyper_dict[i]

    #     # load and print hyper
    #     self.__load_hyper(
    #         hyper=hyper,
    #         baseline_cnn_path=baseline_cnn_path,
    #         debug_mode=debug_mode,
    #     )
    #     self.__print_hyper(hyper)

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
    #     hyper["time.used"] = datetime.now()

    #     self.__training_cur_round(
    #         cur_result_folder=cur_result_folder,
    #         cur_patient=cur_patient,
    #         annotated_slices=annotated_slices,
    #         label_folder=cur_round_folder,
    #     )

    #     # get training time used before save hyper
    #     hyper["time.used"] = datetime.now() - hyper["time.used"]
    #     # save hyper
    #     self.__save_hyper(os.path.join(cur_result_folder, "hyper.json"))

    # in this function, cur round slices have not been added into annotated_slices
    def __select_cur_round_slices(
        self, patient_folder: str, annotated_slices: dict, hyper: NestedDict
    ) -> list:  # return a list of int

        cur_round_slices = []
        candidate_slices = NestedDict()  # {"slice_id": pred_tumor_size}
        cur_round = len(annotated_slices) + 1

        patient = Path(patient_folder).name
        patient = patient[len("patient=") :]

        # get prev round pred and label path
        if cur_round == 1:
            prev_round_pred_folder = Path(patient_folder).parent.parent.parent
            prev_round_pred_folder = os.path.join(
                prev_round_pred_folder,
                "baseline",
                "patients",
                "patient={}".format(patient),
            )
        else:
            prev_round_pred_folder = os.path.join(
                patient_folder, "round={:02d}".format(cur_round - 1)
            )
        pred = g.load_nii(
            os.path.join(prev_round_pred_folder, "pred_gtvs.nii"),
            binary=True,
            out_dim=3,
        )
        label = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)),
            binary=True,
            out_dim=3,
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
        if hyper["select.scenario"] == "equal.divide" and cur_round == 1:
            divided_parts = hyper["select.step"][0] + 1
            candidate_slices = g.get_dict_keys(candidate_slices)
            for i in range(1, divided_parts):
                idx = len(candidate_slices) * i / divided_parts
                idx = round(idx)
                idx = g.check_limit(idx, 1, len(candidate_slices))
                cur_round_slices.append(candidate_slices[idx - 1])

        # "random"
        elif hyper["select.scenario"] == "random":
            cur_round_slices = g.get_dict_keys(candidate_slices)
            random.shuffle(cur_round_slices)

        # (1) "largest"
        # (2) "equal.divide", round >= 2
        else:
            # descrease sort the dict (return a list of tuple)
            candidate_slices = g.sort_dict_by_value(candidate_slices, reverse=True)
            cur_round_slices = g.get_dict_keys(candidate_slices)

        # narrow cur_round_slices based on select.step
        cur_round_slices_num = hyper["select.step"][cur_round - 1]
        if cur_round_slices_num < len(cur_round_slices):
            cur_round_slices = cur_round_slices[:cur_round_slices_num]

        return cur_round_slices

    def __inference_cur_round(self, cur_round_folder: str, hyper: NestedDict):
        cur_round = Path(cur_round_folder).name

        patient = Path(cur_round_folder).parent.name

        score_json_path = Path(cur_round_folder).parent.parent.parent
        score_json_path = os.path.join(score_json_path, "score.json")

        # result contains: "gtvs" "dsc" "msc" "hd95"
        patient_result = self._inference_single_patient(
            patient=patient[len("patient=") :], hyper=hyper
        )

        # save score of cur patient
        score = g.load_json(score_json_path)
        for i in g.METRICS_LIST:
            score[patient][i][cur_round] = patient_result[i]
        g.save_json(score, score_json_path)

        # save pred of cur patient
        for i in ["gtvs"]:  # ["gtvt", "gtvn"]:
            g.save_nii(
                img=patient_result[i],
                save_path=os.path.join(cur_round_folder, "pred_{}.nii".format(i)),
                spacing=g.NII_SPACING,
            )
            g.save_nii(
                img=g.binarize_img(patient_result[i]),
                save_path=os.path.join(
                    cur_round_folder, "pred_{}_binary.nii".format(i)
                ),
                spacing=g.NII_SPACING,
            )

    def __training_cur_round(
        self,
        cur_round_folder: str,
        hyper: NestedDict,
        annotated_slices: dict,
        label_folder: str,
    ):
        g.create_folder(cur_round_folder)

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

        # record time used
        cur_round_time_used = datetime.now()

        # create iDL dataset
        idl_dataset = IDLDataSet(
            patient=patient,
            annotated_slices=annotated_slices,
            label_folder=label_folder,
            pred_folder=pred_folder,
            ignore_other_anotated_slices=False,
            augment_methods=hyper["augment.methods"],
            augment_times=hyper["augment.times"],
            augment_pct=hyper["augment.pct"],
            augment_low_limit=hyper["augment.low.limit"],
            augment_up_limit=hyper["augment.up.limit"],
        )

        # optimize batch size (before create dataloader)
        self._optimize_batch_size(hyper=hyper, dataset=idl_dataset)

        # idl dataloader
        idl_loader = DataLoader(
            dataset=idl_dataset,
            batch_size=hyper["batch.size.actual"],
            shuffle=True,
            num_workers=g.NUM_WORKERS,
        )

        # iter loop
        for cur_iter in tqdm(range(hyper["iter"])):
            hyper["cnn"].train()
            sum_loss = 0
            batch_num = 0

            # freeze layers before iDL
            if hyper["layer.freezing"]:
                if g.used_gpu_count() > 1:
                    # here, hyper["cnn"] is DataParallel, not network itself
                    hyper["cnn"].module.freeze_top()
                else:
                    hyper["cnn"].freeze_top()

            for inputs, labels, weight_map in idl_loader:
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                weight_map = weight_map.to(g.DEVICE)
                labels = labels * weight_map
                outputs = hyper["cnn"](inputs)[3]
                outputs = outputs * weight_map
                loss = hyper["loss.func"](outputs, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                sum_loss += loss.item()
                batch_num += 1

            # cur iter finished
            # update scheduler
            iter_loss = sum_loss / batch_num
            hyper["scheduler"].step(iter_loss)
            # record loss
            loss_dict[
                "iter={:03d}".format((cur_round - 1) * hyper["iter"] + (cur_iter + 1))
            ] = iter_loss

        # current round finished
        # inference
        self.__inference_cur_round(cur_round_folder=cur_round_folder, hyper=hyper)

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
        hyper: NestedDict,
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

        # copy baseline score to idl score
        baseline_score_json_path = Path(patient_folder).parent.parent.parent
        baseline_score_json_path = os.path.join(
            baseline_score_json_path, "baseline", "score.json"
        )
        baseline_score = g.load_json(baseline_score_json_path)
        idl_score_json_path = Path(patient_folder).parent.parent
        idl_score_json_path = os.path.join(idl_score_json_path, "score.json")
        idl_score = g.load_json(idl_score_json_path)
        for i in g.METRICS_LIST:
            idl_score["patient={}".format(patient)][i]["round=00"] = baseline_score[
                "patient={}".format(patient)
            ][i]
        g.save_json(idl_score, idl_score_json_path)

        g.print_line()
        print("patient:", patient)

        annotated_slices = NestedDict()

        # loop through each round
        for cur_round in range(1, len(hyper["select.step"]) + 1):

            cur_round_slices = self.__select_cur_round_slices(
                patient_folder=patient_folder,
                annotated_slices=annotated_slices,
                hyper=hyper,
            )
            if len(cur_round_slices) == 0:
                break

            # start current round
            print("round:", cur_round)

            # add cur_round_slices into annotated_slices BEFORE __training_cur_round()
            annotated_slices["round={:02d}".format(cur_round)] = cur_round_slices

            cur_round_folder = os.path.join(
                patient_folder, "round={:02d}".format(cur_round)
            )
            self.__training_cur_round(
                cur_round_folder=cur_round_folder,
                hyper=hyper,
                annotated_slices=annotated_slices,
                label_folder=g.DATASET_FOLDER,
            )

            # load new lr before next round
            if hyper["lr.reset"]:
                self.__load_next_round_lr(next_round=cur_round + 1)

        # save annotated slices in cur patient folder
        for i in annotated_slices:
            annotated_slices[i] = g.list_to_str(annotated_slices[i])
        g.save_json(
            data=annotated_slices,
            path=os.path.join(
                idl_folder,
                "patients",
                "patient={}".format(patient),
                "annotated_slices.json",
            ),
        )

    def simulation(
        self,
        baseline_id: str,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        for hyper in self._load_group_hyper(g.HYPER_JSON_IDL):

            baseline_cnn_path = g.get_sub_files(
                os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline"),
                key_word=".pt",
                return_full_path=True,
            )[0]

            self.__load_hyper(
                hyper=hyper,
                baseline_cnn_path=baseline_cnn_path,
                debug_mode=debug_mode,
            )

            idl_id = "idl_" + self._init_train_id(
                train_remark=train_remark,
                debug_mode=debug_mode,
                hyper_json_path=g.HYPER_JSON_IDL,
                hyper=hyper,
            )
            g.print_line()
            print(idl_id)
            self.__print_hyper(hyper)

            # create idl result folder
            idl_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, idl_id)
            g.create_folder(idl_folder)

            # save hyper before training
            hyper_save_path = os.path.join(idl_folder, "hyper.json")
            self.__save_hyper(hyper, hyper_save_path)

            # create an empty score.json
            g.save_json(NestedDict(), os.path.join(idl_folder, "score.json"))

            # training start time
            hyper["time.used"] = datetime.now()

            # patient loop
            for patient in hyper["patients"]:

                self.__training_cur_patient(
                    patient=patient,
                    hyper=hyper,
                    idl_folder=idl_folder,
                )

                # reset cnn/optimizer/scheduler before next patient
                if patient != hyper["patients"][-1]:
                    self.__reset_cnn(
                        hyper=hyper,
                        baseline_cnn_path=baseline_cnn_path,
                    )

            # get training time used before save hyper
            hyper["time.used"] = datetime.now() - hyper["time.used"]
            hyper["time.used"] = str(hyper["time.used"]).split(".", 2)[0]
            self.__save_hyper(hyper_save_path)

            self.__record_avg_score(idl_folder)

    def __record_avg_score(self, idl_folder: str):
        score_json_path = os.path.join(idl_folder, "score.json")
        score = g.load_json(score_json_path)
        avg = NestedDict()

        for patient in score:
            for metric in g.METRICS_LIST:
                for cur_round in score[patient][metric]:
                    if avg[metric][cur_round] == {}:
                        avg[metric][cur_round] = []
                    avg[metric][cur_round].append(score[patient][metric][cur_round])

        for metric in g.METRICS_LIST:
            for cur_round in avg[metric]:
                score["avg"][metric][cur_round] = g.get_avg_value(
                    avg[metric][cur_round]
                )
        g.save_json(data=score, path=os.path.join(score_json_path))
