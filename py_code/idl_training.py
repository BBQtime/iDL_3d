import os
import random
import numpy as np
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from loss_func import UnifiedFocalLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import global_elems as g
from idl_dataset import IDLDataSet
from shared_training import SharedTraining
from nested_dict import NestedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import measurements


class IDLTraining(SharedTraining):
    def __load_next_round_lr(self, next_round: int, hyper: NestedDict):
        # hyper["lr"]["init"] is a list of lr of each round
        if next_round > len(hyper["lr"]["init"]):
            next_round = len(hyper["lr"]["init"])

        if g.used_gpu_count() > 1:
            hyper["lr"]["actual"].append(
                hyper["lr"]["init"][next_round - 1] * g.used_gpu_count()
            )
        else:
            hyper["lr"]["actual"].append(hyper["lr"]["init"][next_round - 1])

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr"]["actual"][-1]
        )

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        hyper["scheduler"] = ReduceLROnPlateau(
            optimizer=hyper["optim"],
            mode="min",
            factor=hyper["lr"]["decay.factor"],  # "factor=1" will cause an error
            patience=hyper["lr"]["decay.patience"],
            min_lr=hyper["lr"]["min"],
        )

    # reset cnn/optimizer/scheduler before next patient
    def __reset_cnn(self, hyper: dict, baseline_cnn_path: str):
        # reload cnn
        super()._load_cnn(hyper, baseline_cnn_path)

        # if g.used_gpu_count() > 1:
        #     hyper["lr"]["actual"] = hyper["lr"]["init"][0] * g.used_gpu_count()
        # else:
        #     hyper["lr"]["actual"] = hyper["lr"]["init"][0]

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr"]["actual"][0]
        )

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        hyper["scheduler"] = ReduceLROnPlateau(
            optimizer=hyper["optim"],
            mode="min",
            factor=hyper["lr"]["decay.factor"],  # "factor=1" will cause an error
            patience=hyper["lr"]["decay.patience"],
            min_lr=hyper["lr"]["min"],
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

        # reset lr before next round or not
        hyper["lr"]["reset"] = bool(hyper["lr"]["reset"])

        # min lr (before init lr)
        hyper["lr"]["min"] = float(hyper["lr"]["min"])

        # lr (list)
        # the list of lr is saved in json file as a string, not a list, because:
        # (1) string is easier to read the json file (only one line)
        # (2) a "list" will be recognized as multiple hyper parameters
        hyper["lr"]["init"] = g.str_to_list(hyper["lr"]["init"])
        for i in range(len(hyper["lr"]["init"])):
            hyper["lr"]["init"][i] = float(hyper["lr"]["init"][i])
            hyper["lr"]["init"][i] = g.check_limit(hyper["lr"]["init"][i], 1e-10, None)
            # check min lr, make sure it is lower than any lr in the lr list
            hyper["lr"]["min"] = g.check_limit(
                hyper["lr"]["min"], 0, hyper["lr"]["init"][i]
            )

        hyper["lr"]["actual"] = []
        if g.used_gpu_count() > 1:
            hyper["lr"]["actual"].append(hyper["lr"]["init"][0] * g.used_gpu_count())
        else:
            hyper["lr"]["actual"].append(hyper["lr"]["init"][0])

        # lr decay patience (before shared hyper)
        hyper["lr"]["decay.patience"] = int(hyper["lr"]["decay.patience"])
        g.check_limit(hyper["lr"]["decay.patience"], 1, hyper["iter"])

        # augmentation times
        hyper["augment"]["times"] = int(hyper["augment"]["times"])
        hyper["augment"]["times"] = g.check_limit(hyper["augment"]["times"], 1, None)

        # augmentation percent (based on augment_times)
        hyper["augment"]["pct"] = hyper["augment"]["times"] / (
            hyper["augment"]["times"] + 1
        )

        # freeze layers
        hyper["layer.freezing"] = bool(hyper["layer.freezing"])

        # select.step is saved in json file as a string, not a list, because:
        # (1) it's easier to read the json file (only one line)
        # (2) a "list" will be recognized as multiple hyper parameters,
        # then start multiple training
        for plane in ["transverse", "coronal", "sagittal"]:
            hyper["select.step"][plane] = g.str_to_list(hyper["select.step"][plane])
            for i in range(len(hyper["select.step"][plane])):
                hyper["select.step"][plane][i] = int(hyper["select.step"][plane][i])
                hyper["select.step"][plane][i] = g.check_limit(
                    hyper["select.step"][plane][i], 0, None
                )

        # select scenario
        for plane in ["transverse", "coronal", "sagittal"]:
            hyper["select.scenario"][plane] = str(
                hyper["select.scenario"][plane]
            ).lower()
            if (
                hyper["select.scenario"][plane] != "largest"
                and hyper["select.scenario"][plane] != "gravity.center"
                and hyper["select.scenario"][plane] != "equal.divide"
            ):
                hyper["select.scenario"][plane] = "random"

        # weight map parameters
        hyper["weight"]["background"] = float(hyper["weight"]["background"])
        if hyper["weight"]["background"] > 1:
            hyper["weight"]["background"] = 1

        hyper["weight"]["slice"] = float(hyper["weight"]["slice"])
        if hyper["weight"]["slice"] < hyper["weight"]["background"]:
            hyper["weight"]["slice"] = hyper["weight"]["background"]

        hyper["weight"]["annotation"] = float(hyper["weight"]["annotation"])
        if hyper["weight"]["annotation"] < hyper["weight"]["slice"]:
            hyper["weight"]["annotation"] = hyper["weight"]["slice"]

        hyper["weight"]["distance.step"] = int(hyper["weight"]["distance.step"])
        if hyper["weight"]["distance.step"] < 1:
            hyper["weight"]["distance.step"] = 1

        hyper["weight"]["prev.round.decay"] = float(hyper["weight"]["prev.round.decay"])
        if hyper["weight"]["prev.round.decay"] > 1:
            hyper["weight"]["prev.round.decay"] = 1.0

        # load patients
        hyper["patients"] = self.__load_dataset(debug_mode)

        # load shared hyper
        super()._load_hyper(
            hyper=hyper,
            exist_cnn_path=baseline_cnn_path,
        )

        # run this after shared hyper loaded, loss parameters are needed
        hyper["loss"]["func"] = UnifiedFocalLoss(
            asym=hyper["loss"]["asym"],
            weight=hyper["loss"]["weight"],
            delta=hyper["loss"]["delta"],
            gamma=hyper["loss"]["gamma"],
            gtvt_only=True,
        ).to(g.DEVICE)

    def __load_dataset(self, debug_mode: bool = False):
        json_data = g.load_json(g.DATASET_SPLITTING_JSON)
        test_patients = g.str_to_list(json_data["test.patients"])

        # debug mode, only 1 or 2 patients
        if debug_mode:
            test_patients = test_patients[:1]
        return test_patients

    def __get_simple_hyper(self, hyper: NestedDict) -> NestedDict:
        simple_hyper = NestedDict()
        for i in hyper:
            if i == "lr":
                simple_hyper[i] = hyper[i].copy()
                for k in ["init", "actual"]:
                    simple_hyper[i][k] = g.list_to_str(simple_hyper[i][k])

            elif i == "patients":
                simple_hyper[i] = len(hyper[i])

            elif i == "select.step":
                simple_hyper[i] = hyper[i].copy()
                for plane in simple_hyper[i]:
                    simple_hyper[i][plane] = g.list_to_str(simple_hyper[i][plane])

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
    #     hyper["time.spent"] = datetime.now()

    #     self.__training_cur_round(
    #         cur_result_folder=cur_result_folder,
    #         cur_patient=cur_patient,
    #         annotated_slices=annotated_slices,
    #         label_folder=cur_round_folder,
    #     )

    #     # get training time spent before save hyper
    #     hyper["time.spent"] = datetime.now() - hyper["time.spent"]
    #     # save hyper
    #     self.__save_hyper(os.path.join(cur_result_folder, "hyper.json"))

    # in this function, cur round slices have not been added into annotated_slices
    def __select_cur_round_slices(
        self,
        annotated_slices: dict,
        hyper: NestedDict,
        patient_folder: str,
        baseline_epoch_folder: str,
    ) -> list:  # return a list of int

        cur_round_slices = NestedDict()
        for plane in ["transverse", "coronal", "sagittal"]:
            cur_round_slices[plane] = []

        cur_round = max(
            len(annotated_slices["transverse"]),
            len(annotated_slices["coronal"]),
            len(annotated_slices["sagittal"]),
        )
        cur_round += 1

        patient = Path(patient_folder).name
        patient = patient[len("patient=") :]

        # get prev round folder path
        if cur_round == 1:
            prev_round_pred_folder = os.path.join(
                baseline_epoch_folder,
                "patients",
                "patient={}".format(patient),
            )
        else:
            prev_round_pred_folder = os.path.join(
                patient_folder, "round={:02d}".format(cur_round - 1)
            )

        # load gtvt pred/label
        pred = g.load_nii(
            os.path.join(prev_round_pred_folder, "pred_gtvt.nii"),
            binary=True,
        )
        label = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )
        target = np.maximum(pred, label)
        target_center = measurements.center_of_mass(target)

        # select slices through each plane
        for plane in ["transverse", "coronal", "sagittal"]:

            # skip cur plane if no slice needs to be selected
            if len(hyper["select.step"][plane]) < cur_round:
                continue

            candidate_slices = dict()
            cur_plane_annotated_slices = g.dict_to_list(annotated_slices[plane])

            # go through pred and record tumor size
            if plane == "transverse":
                slice_counts = target.shape[0]
            elif plane == "coronal":
                slice_counts = target.shape[1]
            elif plane == "sagittal":
                slice_counts = target.shape[2]

            for cur_slice in range(slice_counts):
                # skip slice that already been annotated
                if cur_slice in cur_plane_annotated_slices:
                    continue
                else:
                    if plane == "transverse":
                        cur_slice_target_volume = target[cur_slice, :, :].sum()
                    elif plane == "coronal":
                        cur_slice_target_volume = target[:, cur_slice, :].sum()
                    elif plane == "sagittal":
                        cur_slice_target_volume = target[:, :, cur_slice].sum()
                    # add slice with target (pred or label) into candidates
                    if cur_slice_target_volume > 0:
                        candidate_slices[cur_slice] = cur_slice_target_volume

            # "largest"
            if hyper["select.scenario"][plane] == "largest":
                # descrease sort the dict (return a list of tuple)
                candidate_slices = g.sort_dict_by_value(candidate_slices, reverse=True)
                cur_round_slices[plane] = g.get_dict_keys(candidate_slices)

            # "gravity.center", round =1
            elif hyper["select.scenario"][plane] == "gravity.center" and cur_round == 1:
                if plane == "transverse":
                    cur_round_slices[plane].append(round(target_center[0]))
                elif plane == "coronal":
                    cur_round_slices[plane].append(round(target_center[1]))
                elif plane == "sagittal":
                    cur_round_slices[plane].append(round(target_center[2]))

            # "equal.divide", round = 1
            elif hyper["select.scenario"][plane] == "equal.divide" and cur_round == 1:
                divided_parts = hyper["select.step"][plane][0] + 1
                candidate_slices = g.get_dict_keys(candidate_slices)
                for part in range(1, divided_parts):
                    idx = len(candidate_slices) * part / divided_parts
                    idx = round(idx)
                    idx = g.check_limit(idx, 1, len(candidate_slices))
                    cur_round_slices[plane].append(candidate_slices[idx - 1])

            # (1) "random"
            # (2) "gravity.center", round >= 2
            # (3) "equal.divide", round >= 2
            else:
                cur_round_slices[plane] = g.get_dict_keys(candidate_slices)
                random.shuffle(cur_round_slices[plane])

            # narrow cur_round_slices based on select.step
            if hyper["select.scenario"][plane] == "gravity.center" and cur_round == 1:
                cur_round_slices_count = 1
            else:
                cur_round_slices_count = hyper["select.step"][plane][cur_round - 1]
            if cur_round_slices_count < len(cur_round_slices[plane]):
                cur_round_slices[plane] = cur_round_slices[plane][
                    :cur_round_slices_count
                ]

            # add cur_round_slices into annotated_slices
            annotated_slices[plane][
                "round={:02d}".format(cur_round)
            ] = cur_round_slices[plane]

        return cur_round_slices

    def __inference_cur_round(self, cur_round_folder: str, hyper: NestedDict):
        cur_round = Path(cur_round_folder).name

        patient = Path(cur_round_folder).parent.name

        # result structure: gtvs/gtvt/gvtn → pred/dsc/msd/hd95
        patient_result = self._inference_single_patient(
            patient=patient[len("patient=") :], hyper=hyper
        )

        # save score of cur patient
        idl_folder = Path(cur_round_folder).parent.parent.parent
        for gtv in ["gtvt"]:  # ["gtvs", "gtvt", "gtvn"]:
            score_json_path = os.path.join(idl_folder, "score_{}.json".format(gtv))
            score = g.load_json(score_json_path)
            for metric_type in g.METRICS_LIST:
                score[patient][metric_type][cur_round] = patient_result[gtv][
                    metric_type
                ]
            g.save_json(score, score_json_path)

        # save pred of cur patient
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            g.save_nii(
                img=patient_result[gtv]["pred"],
                save_path=os.path.join(cur_round_folder, "pred_{}.nii".format(gtv)),
                spacing=g.NII_SPACING,
            )
            # g.save_nii(
            #     img=g.binarize_img(patient_result[gtv]["pred"]),
            #     save_path=os.path.join(
            #         cur_round_folder, "pred_{}_binary.nii".format(gtv)
            #     ),
            #     spacing=g.NII_SPACING,
            # )

    def __training_cur_round(
        self,
        cur_round_folder: str,
        baseline_epoch_folder: str,
        label_folder: str,
        hyper: NestedDict,
        annotated_slices: dict,
    ):
        g.create_folder(cur_round_folder)

        cur_round = Path(cur_round_folder).name
        cur_round = int(cur_round[len("round=") :])

        patient = Path(cur_round_folder).parent.name
        patient = patient[len("patient=") :]

        idl_folder = Path(cur_round_folder).parent.parent.parent
        loss_json_path = os.path.join(Path(cur_round_folder).parent, "loss.json")
        loss_dict = g.load_json(loss_json_path)

        if cur_round == 1:
            pred_folder = os.path.join(
                baseline_epoch_folder, "patients", "patient={}".format(patient)
            )
        else:
            pred_folder = os.path.join(
                Path(cur_round_folder).parent, "round={:02d}".format(cur_round - 1)
            )

        # record current round time spent
        cur_round_time_spent = datetime.now()

        # create iDL dataset
        idl_dataset = IDLDataSet(
            patient=patient,
            annotated_slices=annotated_slices,
            label_folder=label_folder,
            pred_folder=pred_folder,
            augment=hyper["augment"],
            weight=hyper["weight"],
        )

        # optimize batch size (before create dataloader)
        self._optimize_batch_size(hyper=hyper, dataset=idl_dataset)

        # idl dataloader
        idl_loader = DataLoader(
            dataset=idl_dataset,
            batch_size=hyper["batch.size"]["actual"],
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
                outputs = hyper["cnn"](inputs)[3]
                loss = hyper["loss"]["func"](outputs, labels, weight_map)
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
            # save loss and update loss figure after every iter, if there is only one patient
            patient_folder_list = g.get_sub_folders(
                os.path.join(idl_folder, "patients")
            )
            if len(patient_folder_list) <= 1:
                g.save_json(loss_dict, loss_json_path)
                self.__draw_loss_fig(idl_folder)

        # current round finished
        # inference
        self.__inference_cur_round(cur_round_folder=cur_round_folder, hyper=hyper)

        # save time spent
        cur_round_time_spent = datetime.now() - cur_round_time_spent
        round_str = "round={:02d}".format(cur_round)
        if hyper["time.spent"]["avg"][round_str] == {}:
            hyper["time.spent"]["avg"][round_str] = cur_round_time_spent
        else:
            hyper["time.spent"]["avg"][round_str] += cur_round_time_spent

        # save loss
        g.save_json(loss_dict, loss_json_path)

    def __training_cur_patient(
        self,
        patient: str,
        baseline_epoch_folder: str,
        idl_folder: str,
        hyper: NestedDict,
    ):
        # create current patient folder
        patient_folder = os.path.join(
            idl_folder, "patients", "patient={}".format(patient)
        )
        g.create_folder(patient_folder)
        # create an empty loss.json
        g.save_json(NestedDict(), os.path.join(patient_folder, "loss.json"))

        # initialize idl score (copy from baseline)
        baseline_score = g.load_json(
            os.path.join(baseline_epoch_folder, "score_test.json")
        )
        for gtv in ["gtvt"]:  # ["gtvs", "gtvt", "gtvn"]:
            idl_score_json_path = os.path.join(idl_folder, "score_{}.json".format(gtv))
            idl_score = g.load_json(idl_score_json_path)
            for metric_type in g.METRICS_LIST:
                idl_score["patient={}".format(patient)][metric_type][
                    "round=00"
                ] = baseline_score["patient={}".format(patient)][gtv][metric_type]
            g.save_json(idl_score, idl_score_json_path)

        g.print_line()
        print("patient:", patient)

        annotated_slices = NestedDict()

        # loop through each round
        max_round = max(
            len(hyper["select.step"]["transverse"]),
            len(hyper["select.step"]["coronal"]),
            len(hyper["select.step"]["sagittal"]),
        )
        for cur_round in range(1, max_round + 1):

            # cur round slices are add into annotated_slices in this function
            cur_round_slices = self.__select_cur_round_slices(
                annotated_slices=annotated_slices,
                hyper=hyper,
                patient_folder=patient_folder,
                baseline_epoch_folder=baseline_epoch_folder,
            )

            # no slice needs to be annotated in cur round
            if (
                len(cur_round_slices["transverse"]) == 0
                and len(cur_round_slices["coronal"]) == 0
                and len(cur_round_slices["sagittal"]) == 0
            ):
                break

            # start current round
            print("round:", cur_round)

            cur_round_folder = os.path.join(
                patient_folder, "round={:02d}".format(cur_round)
            )
            self.__training_cur_round(
                cur_round_folder=cur_round_folder,
                baseline_epoch_folder=baseline_epoch_folder,
                label_folder=g.DATASET_FOLDER,
                hyper=hyper,
                annotated_slices=annotated_slices,
            )

            if cur_round == max_round:
                break

            # load new lr before next round
            if hyper["lr"]["reset"]:
                self.__load_next_round_lr(cur_round + 1, hyper)

        # draw avg loss of all trained patients
        self.__draw_loss_fig(idl_folder)

        # save annotated slices in cur patient folder
        for plane in ["transverse", "coronal", "sagittal"]:
            for cur_round in annotated_slices[plane]:
                annotated_slices[plane][cur_round] = g.list_to_str(
                    annotated_slices[plane][cur_round]
                )
        g.save_json(
            data=annotated_slices,
            path=os.path.join(
                idl_folder,
                "patients",
                "patient={}".format(patient),
                "annotated_slices.json",
            ),
        )

    def draw_loss_fig(self, baseline_id: str, idl_id: str):
        idl_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, idl_id)
        self.__draw_loss_fig(idl_folder)

    def __draw_loss_fig(self, idl_folder: str):
        # avg loss dict
        avg_loss = NestedDict()
        for cur_patient_folder in g.get_sub_folders(
            os.path.join(idl_folder, "patients"), return_full_path=True
        ):
            cur_patient_loss = g.load_json(
                os.path.join(cur_patient_folder, "loss.json")
            )
            if len(avg_loss) == 0:
                for i in cur_patient_loss:
                    avg_loss[i] = [cur_patient_loss[i]]
            else:
                for i in avg_loss:
                    avg_loss[i].append(cur_patient_loss[i])

        for i in avg_loss:
            avg_loss[i] = g.get_avg_value(avg_loss[i])

        avg_loss = g.dict_to_list(avg_loss)

        # draw figure
        plt.figure().clear()
        plt.plot(range(1, len(avg_loss) + 1), avg_loss, label="loss")
        plt.legend()
        plt.savefig(os.path.join(idl_folder, "loss.png"))

    def simulation(
        self,
        baseline_id: str,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        for hyper in self._load_group_hyper(g.HYPER_JSON_IDL):

            idl_id = "idl_" + self._init_train_id(
                train_remark=train_remark,
                debug_mode=debug_mode,
                hyper_json_path=g.HYPER_JSON_IDL,
                hyper=hyper,
            )
            g.print_line()
            print(idl_id)

            # use first fold folder
            baseline_fold_folder = g.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline"),
                key_word="fold=",
                return_full_path=True,
            )[0]
            # use first epoch folder
            baseline_epoch_folder = g.get_sub_folders(
                baseline_fold_folder, "epoch=", return_full_path=True
            )[0]
            baseline_cnn_path = os.path.join(baseline_epoch_folder, "cnn.pt")

            self.__load_hyper(
                hyper=hyper,
                baseline_cnn_path=baseline_cnn_path,
                debug_mode=debug_mode,
            )
            g.print_line()
            self.__print_hyper(hyper)

            # create idl result folder
            idl_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, idl_id)
            g.create_folder(idl_folder)

            # save hyper before training
            hyper_save_path = os.path.join(idl_folder, "hyper.json")
            self.__save_hyper(hyper, hyper_save_path)

            # create an empty score json files
            for gtv in ["gtvt"]:  # ["gtvs", "gtvt", "gtvn"]:
                g.save_json(
                    NestedDict(), os.path.join(idl_folder, "score_{}.json".format(gtv))
                )

            # training start time
            hyper["time.spent"]["total"] = datetime.now()

            # patient loop
            for patient in hyper["patients"]:
                self.__training_cur_patient(
                    patient=patient,
                    hyper=hyper,
                    baseline_epoch_folder=baseline_epoch_folder,
                    idl_folder=idl_folder,
                )

                # reset cnn/optimizer/scheduler before next patient
                if patient != hyper["patients"][-1]:
                    self.__reset_cnn(
                        hyper=hyper,
                        baseline_cnn_path=baseline_cnn_path,
                    )

            # record total time spent
            hyper["time.spent"]["total"] = datetime.now() - hyper["time.spent"]["total"]
            hyper["time.spent"]["total"] = str(hyper["time.spent"]["total"]).split(
                ".", 2
            )[0]

            # record avg time spent per patient
            for cur_round in hyper["time.spent"]["avg"]:
                hyper["time.spent"]["avg"][cur_round] /= len(hyper["patients"])
                hyper["time.spent"]["avg"][cur_round] = str(
                    hyper["time.spent"]["avg"][cur_round]
                ).split(".", 2)[0]

            self.__save_hyper(hyper, hyper_save_path)

            self.__record_avg_score(idl_folder)

    def __record_avg_score(self, idl_folder: str):
        for gtv in ["gtvt"]:  # ["gtvs", "gtvt", "gtvn"]:
            score_json_path = os.path.join(idl_folder, "score_{}.json".format(gtv))
            score = g.load_json(score_json_path)
            avg = NestedDict()

            # add all patients score in to a list
            for patient in score:
                for metric in g.METRICS_LIST:
                    for cur_round in score[patient][metric]:
                        if avg[metric][cur_round] == {}:
                            avg[metric][cur_round] = []
                        avg[metric][cur_round].append(score[patient][metric][cur_round])

            # calculate avg score
            for metric in g.METRICS_LIST:
                for cur_round in avg[metric]:
                    score["avg"][metric][cur_round] = g.get_avg_value(
                        avg[metric][cur_round]
                    )
            g.save_json(data=score, path=os.path.join(score_json_path))
