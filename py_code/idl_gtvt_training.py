import os
import random
import statistics
import numpy as np
from datetime import datetime
from pathlib import Path
from loss_func import UnifiedFocalLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from custom import Global as g
from idl_gtvt_dataset import IDLGTVtDataSet
from training import Training
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import measurements
from custom import Dict
from custom import List
from custom import Json
from custom import GPU
from custom import Nii
from custom import Folder
from custom import Value
from custom import Explorer


class IDLGTVtTraining(Training):
    def __load_next_round_lr(self, next_round: int, hyper: Dict):
        # hyper["lr"] is a list of lr of each round
        if next_round > len(hyper["lr"]):
            next_round = len(hyper["lr"])

        if GPU.used_count() > 1:
            hyper["lr.actual"].append(hyper["lr"][next_round - 1] * GPU.used_count())
        else:
            hyper["lr.actual"].append(hyper["lr"][next_round - 1])

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr.actual"][-1]
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

    # reset cnn/optimizer/scheduler before next patient
    def __reset_cnn(self, hyper: dict, baseline_cnn_path: str):
        # reload cnn
        self._load_cnn(hyper=hyper, cnn_path=baseline_cnn_path)

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr.actual"][0]
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

    def _load_hyper(
        self, hyper: Dict, baseline_cnn_path: str, debug_mode: bool = False
    ):
        # iter
        if debug_mode:
            # at least 2 iters to compare loss difference
            hyper["iter"] = 2
        else:
            hyper["iter"] = Value.limit_range(hyper["iter"], (1, None))

        # lr
        # lr is saved in json file as a string, not a list, because:
        # (1) string is easier to read than list in json file (only one line)
        # (2) a "list" will be recognized as multiple trainings
        hyper["lr"] = List(hyper["lr"])
        for i in range(len(hyper["lr"])):
            hyper["lr"][i] = float(hyper["lr"][i])
            hyper["lr"][i] = Value.limit_range(hyper["lr"][i], (g.EPS, 1))
            # check min lr, make sure it is lower than any lr in the lr list
            hyper["lr.min"] = Value.limit_range(
                hyper["lr.min"], (g.EPS, hyper["lr"][i])
            )

        # actual lr
        hyper["lr.actual"] = List()
        if GPU.used_count() > 1:
            hyper["lr.actual"].append(hyper["lr"][0] * GPU.used_count())
        else:
            hyper["lr.actual"].append(hyper["lr"][0])

        # lr decay patience (before shared hyper)
        hyper["lr.decay.patience"] = Value.limit_range(
            hyper["lr.decay.patience"], (1, hyper["iter"])
        )

        # augmentation times
        hyper["augment.times"] = Value.limit_range(hyper["augment.times"], (1, None))

        # augmentation percent (based on augment_times)
        hyper["augment.pct"] = hyper["augment.times"] / (hyper["augment.times"] + 1)

        # select step
        # select.step is saved in json file as a string, not a list, because:
        # (1) string is easier to read than list in json file (only one line)
        # (2) a "list" will be recognized as multiple trainings
        for plane in ["transverse", "coronal", "sagittal"]:
            plane = "select.step.{}".format(plane)
            hyper[plane] = List(hyper[plane])
            for i in range(len(hyper[plane])):
                hyper[plane][i] = int(hyper[plane][i])
                hyper[plane][i] = Value.limit_range(hyper[plane][i], (0, None))

        # select scenario
        if (
            hyper["select.scenario"] != "largest"
            and hyper["select.scenario"] != "gravity.center"
            and hyper["select.scenario"] != "equal.divide"
        ):
            hyper["select.scenario"] = "random"

        # weight map parameters
        hyper["weight.background"] = Value.limit_range(
            hyper["weight.background"], (0.0, 1.0)
        )
        hyper["weight.slice"] = Value.limit_range(
            hyper["weight.slice"], (hyper["weight.background"], None)
        )
        hyper["weight.fp.fn"] = Value.limit_range(
            hyper["weight.fp.fn"], (hyper["weight.slice"], None)
        )
        hyper["weight.distance.step"] = Value.limit_range(
            hyper["weight.distance.step"], (1, None)
        )
        hyper["weight.prev.round.decay"] = Value.limit_range(
            hyper["weight.prev.round.decay"], (0.0, 1.0)
        )

        # load patients
        hyper["patients"] = self._load_dataset(debug_mode)

        # load shared hyper
        super()._load_hyper(hyper=hyper, cnn_path=baseline_cnn_path)

        # run this after shared hyper loaded, because loss parameters are needed
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
            training="idl_gtvt",
        ).to(g.DEVICE)

    def _load_dataset(self, debug_mode: bool = False):
        json_data = Json.load(g.DATASET_SPLIT_JSON_PATH)
        test_patients = List(json_data["test.set"])

        # debug mode, only 1 or 2 patients
        if debug_mode:
            test_patients = test_patients[:2]

        return test_patients

    def __get_simple_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for cur_key in hyper:
            if cur_key == "lr" or cur_key == "lr.actual":
                simple_hyper[cur_key] = hyper[cur_key].to_str()

            elif cur_key == "patients":
                simple_hyper[cur_key] = len(hyper[cur_key])

            elif "select.step" in cur_key:
                simple_hyper[cur_key] = hyper[cur_key].to_str()

            else:
                simple_hyper[cur_key] = hyper[cur_key]
        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._save_hyper(simple_hyper, json_path)

    # def real_training(
    #     self,
    #     baseline_id: str,
    #     idl_results_folder: str,
    #     idl_gtvt_id: str,
    #     cur_patient: str,
    #     cur_round: int,
    #     debug_mode: bool = False,
    # ):
    #     self._idl_gtvt_id = idl_gtvt_id

    #     # get baseline cnn and hyper path
    #     baseline_cnn_path, baseline_hyper_path = self.__get_baseline_paths(baseline_id)
    #     print("")
    #     print(baseline_cnn_path)
    #     # load hypers
    #     idl_hyper_dict = Json.load(g.HYPER_JSON_PATH_IDL_GTVT)
    #     baseline_hyper_dict = Json.load(baseline_hyper_path)

    #     # make sure all hypers are unique, no arrangement
    #     hyper = Dict()
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
    #     self._print_hyper(hyper)

    #     # check if result folder exist
    #     cur_result_folder = os.path.join(idl_results_folder, self._idl_id)
    #     if not os.path.exists(cur_result_folder):
    #         print("IDLGTVtTraining.real_training(): iDL result folder doesn't exist")

    #     # create json file to save train loss
    #     train_loss_dict = Dict()
    #     train_loss_dict["iter"] = Dict()
    #     Json.save(
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
    #     annotated_slices = Dict()
    #     annotated_slices["round=01"] = List()  # doesn't matter what the dict key is
    #     for file_name in Explorer.get_sub_files(cur_round_folder, key_word="_label.npy"):
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
    #     self._save_hyper(os.path.join(cur_result_folder, "hyper.json"))

    # in this function, cur round slices have not been added into annotated_slices
    def __select_cur_round_slices(
        self,
        annotated_slices: Dict,
        hyper: Dict,
        patient_folder: str,
    ) -> list:  # return a list of int

        cur_round_slices = Dict()
        for plane in ["transverse", "coronal", "sagittal"]:
            cur_round_slices[plane] = List()

        cur_round = max(
            len(annotated_slices["transverse"]),
            len(annotated_slices["coronal"]),
            len(annotated_slices["sagittal"]),
        )
        cur_round += 1

        patient = Path(patient_folder).name
        patient = patient[len("patient=") :]

        label = Nii.load(
            os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )
        # label_center: (d,h,w)
        label_center = measurements.center_of_mass(label)

        # select slices through each plane
        for plane in ["transverse", "coronal", "sagittal"]:

            # skip cur plane if no slice needs to be selected
            if len(hyper["select.step.{}".format(plane)]) < cur_round:
                continue

            candidate_slices = Dict()
            cur_plane_annotated_slices = annotated_slices[plane].to_list()

            # go through pred and record tumor size
            if plane == "transverse":
                slice_counts = label.shape[0]
            elif plane == "coronal":
                slice_counts = label.shape[1]
            elif plane == "sagittal":
                slice_counts = label.shape[2]

            for cur_slice in range(slice_counts):
                # skip slice that already been annotated
                if cur_slice in cur_plane_annotated_slices:
                    continue
                else:
                    if plane == "transverse":
                        cur_slice_tumor_size = label[cur_slice, :, :].sum()
                    elif plane == "coronal":
                        cur_slice_tumor_size = label[:, cur_slice, :].sum()
                    elif plane == "sagittal":
                        cur_slice_tumor_size = label[:, :, cur_slice].sum()
                    # add slice with target (pred or label) into candidates
                    if cur_slice_tumor_size > 0:
                        candidate_slices[cur_slice] = cur_slice_tumor_size

            # "largest"
            if hyper["select.scenario"] == "largest":
                # descrease sort the dict (return a list of tuple)
                candidate_slices = candidate_slices.sort_by_value(reverse=True)
                cur_round_slices[plane] = candidate_slices.keys()

            # "gravity.center", round = 1
            elif hyper["select.scenario"] == "gravity.center" and cur_round == 1:
                if plane == "transverse":
                    cur_round_slices[plane].append(round(label_center[0]))
                elif plane == "coronal":
                    cur_round_slices[plane].append(round(label_center[1]))
                elif plane == "sagittal":
                    cur_round_slices[plane].append(round(label_center[2]))

            # "equal.divide", round = 1
            elif hyper["select.scenario"] == "equal.divide" and cur_round == 1:
                divided_parts = hyper["select.step.{}".format(plane)][0] + 1
                candidate_slices = candidate_slices.keys()
                for part in range(1, divided_parts):
                    idx = len(candidate_slices) * part / divided_parts
                    idx = round(idx)
                    idx = Value.limit_range(idx, (1, len(candidate_slices)))
                    cur_round_slices[plane].append(candidate_slices[idx - 1])

            # (1) "random"
            # (2) "gravity.center", round >= 2
            # (3) "equal.divide", round >= 2
            else:
                cur_round_slices[plane] = candidate_slices.keys()
                cur_round_slices[plane].shuffle()

            # narrow cur_round_slices based on select.step
            if hyper["select.scenario"] == "gravity.center" and cur_round == 1:
                cur_round_slices_count = 1
            else:
                cur_round_slices_count = hyper["select.step.{}".format(plane)][
                    cur_round - 1
                ]
            if cur_round_slices_count < len(cur_round_slices[plane]):
                cur_round_slices[plane] = cur_round_slices[plane][
                    :cur_round_slices_count
                ]

            # add cur_round_slices into annotated_slices
            annotated_slices[plane][
                "round={:02d}".format(cur_round)
            ] = cur_round_slices[plane]

        return cur_round_slices

    def __get_masked_label(self, cur_round_folder: str):
        cur_round = Path(cur_round_folder).name
        patient_folder = Path(cur_round_folder).parent
        patient = patient_folder.name
        # change "patient=123" into "123"
        patient = patient[len("patient=") :]

        label = Nii.load(
            os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )

        annotated_slices = Json.load(
            os.path.join(patient_folder, "annotated_slices.json")
        )

        # annotated slice mask
        slice_mask = Dict()

        # loop through each plane
        for plane in ["transverse", "coronal", "sagittal"]:
            slice_mask[plane] = np.zeros(label.shape, dtype=np.float32)

            # loop through each round
            for cur_round in annotated_slices[plane]:

                # str to list
                annotated_slices[plane][cur_round] = List(
                    annotated_slices[plane][cur_round]
                )
                # current step
                for cur_slice in annotated_slices[plane][cur_round]:
                    # change slice id from str into int
                    cur_slice = int(cur_slice)
                    if plane == "transverse":
                        slice_mask[plane][cur_slice, :, :] = np.ones_like(
                            slice_mask[plane][0, :, :]
                        )
                    elif plane == "coronal":
                        slice_mask[plane][:, cur_slice, :] = np.ones_like(
                            slice_mask[plane][:, 0, :]
                        )
                    elif plane == "sagittal":
                        slice_mask[plane][:, :, cur_slice] = np.ones_like(
                            slice_mask[plane][:, :, 0]
                        )

        # combine slice_mask on 3 planes
        slice_mask = np.maximum(
            np.maximum(slice_mask["transverse"], slice_mask["coronal"]),
            slice_mask["sagittal"],
        )
        if 0:
            Nii.save(
                slice_mask,
                os.path.join(g.PROJ_PATH, "debug", "slice_mask_post_processing.nii"),
            )

        label *= slice_mask
        if 0:
            Nii.save(
                label,
                os.path.join(g.PROJ_PATH, "debug", "label_post_processing.nii"),
            )

        return label

    def __inference_cur_round(self, cur_round_folder: str, hyper: Dict):
        cur_round = Path(cur_round_folder).name

        patient = Path(cur_round_folder).parent.name

        # get annotation for post processing
        masked_label = self.__get_masked_label(cur_round_folder)

        # result structure: gtvt: {pred, dsc, msd, hd95}
        patient_result = self._patient_inference(
            patient=patient[len("patient=") :],
            hyper=hyper,
            inference_type="idl_gtvt",
            masked_label=masked_label,
        )

        # save score of cur patient
        idl_gtvt_folder = Path(cur_round_folder).parent.parent.parent
        score_json_path = os.path.join(idl_gtvt_folder, "score.json")
        score = Json.load(score_json_path)
        for metric in g.METRICS:
            score[patient][metric][cur_round] = patient_result["gtvt"][metric]
        Json.save(score, score_json_path)

        # save pred of cur patient
        Nii.save(
            img=patient_result["gtvt"]["pred"],
            path=os.path.join(cur_round_folder, "pred_gtvt.nii"),
            spacing=g.NII_SPACING,
        )

    def __training_cur_round(
        self,
        cur_round_folder: str,
        baseline_epoch_folder: str,
        label_folder: str,
        hyper: Dict,
        annotated_slices: Dict,
    ):
        Folder.create(cur_round_folder)

        cur_round = Path(cur_round_folder).name
        cur_round = int(cur_round[len("round=") :])

        patient_folder = Path(cur_round_folder).parent
        patient = patient_folder.name[len("patient=") :]

        idl_gtvt_folder = patient_folder.parent.parent
        loss_json_path = os.path.join(patient_folder, "loss.json")
        loss_dict = Json.load(loss_json_path)

        if cur_round == 1:
            pred_folder = os.path.join(
                baseline_epoch_folder,
                "baseline",
                "patients",
                "patient={}".format(patient),
            )
        else:
            pred_folder = os.path.join(
                patient_folder, "round={:02d}".format(cur_round - 1)
            )

        # record current round time spent
        cur_round_time_spent = datetime.now()

        # create iDL dataset
        augment = Dict()
        augment["methods"] = hyper["augment.methods"]
        augment["pct"] = hyper["augment.pct"]
        augment["times"] = hyper["augment.times"]
        augment["min"] = hyper["augment.min"]
        augment["max"] = hyper["augment.max"]
        weight = Dict()
        weight["background"] = hyper["weight.background"]
        weight["distance.step"] = hyper["weight.distance.step"]
        weight["fp.fn"] = hyper["weight.fp.fn"]
        weight["prev.round.decay"] = hyper["weight.prev.round.decay"]
        weight["slice"] = hyper["weight.slice"]
        idl_gtvt_dataset = IDLGTVtDataSet(
            patient=patient,
            annotated_slices=annotated_slices,
            label_folder=label_folder,
            pred_folder=pred_folder,
            augment=augment,
            weight=weight,
        )

        # optimize batch size (before create dataloader)
        self._optimize_batch_size(dataset=idl_gtvt_dataset, hyper=hyper)

        # idl gtvt dataloader
        idl_gtvt_loader = DataLoader(
            dataset=idl_gtvt_dataset,
            batch_size=hyper["batch.size.actual"],
            shuffle=True,
            num_workers=g.NUM_WORKERS,
        )

        # iter loop
        for cur_iter in tqdm(range(hyper["iter"])):
            hyper["cnn"].train()
            sum_loss = 0
            num_batches = 0

            # freeze layers before iDL
            if hyper["layer.freezing"]:
                if GPU.used_count() > 1:
                    # here, hyper["cnn"] is DataParallel, not network itself
                    hyper["cnn"].module.freeze_top()
                else:
                    hyper["cnn"].freeze_top()

            # here, labels only have 2 channels: background and gtvt, No gtvn
            for multimodal_imgs, labels, weight_map in idl_gtvt_loader:
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                multimodal_imgs = multimodal_imgs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                weight_map = weight_map.to(g.DEVICE)
                preds = hyper["cnn"](multimodal_imgs)
                loss = hyper["loss.func"](preds, labels, weight_map)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                sum_loss += loss.item()
                num_batches += 1

            # cur iter finished
            # update scheduler
            iter_loss = sum_loss / num_batches
            hyper["scheduler"].step(iter_loss)

            # record loss
            loss_dict[
                "iter={:03d}".format((cur_round - 1) * hyper["iter"] + (cur_iter + 1))
            ] = iter_loss
            # save loss and update loss figure after every iter, if there is only one patient
            patient_folder_list = Explorer.get_sub_folders(
                os.path.join(idl_gtvt_folder, "patients")
            )
            if len(patient_folder_list) <= 1:
                Json.save(loss_dict, loss_json_path)
                self._plot_loss_fig(idl_gtvt_folder)

        # current round idl finished
        # save cnn
        cnn_save_path = os.path.join(
            cur_round_folder, Path(cur_round_folder).name + ".pt"
        )
        self._save_cnn(hyper, cnn_save_path)

        # save annotated_slices dict before inference, because masked_label needs it
        # copy a new dict to avoid changing origin annotated_slices dict
        annotated_slices_copy = annotated_slices.copy()
        for plane in ["transverse", "coronal", "sagittal"]:
            for round_key in annotated_slices_copy[plane]:
                annotated_slices_copy[plane][round_key] = annotated_slices_copy[plane][
                    round_key
                ].to_str()
        Json.save(
            data=annotated_slices_copy,
            path=os.path.join(patient_folder, "annotated_slices.json"),
        )

        # inference
        self.__inference_cur_round(cur_round_folder=cur_round_folder, hyper=hyper)

        # save time spent
        cur_round_time_spent = datetime.now() - cur_round_time_spent
        key_name = "time.spent.avg.round={:02d}".format(cur_round)
        if hyper[key_name] == {}:
            hyper[key_name] = cur_round_time_spent
        else:
            hyper[key_name] += cur_round_time_spent

        # save loss
        Json.save(loss_dict, loss_json_path)

    def __training_cur_patient(
        self,
        patient: str,
        baseline_epoch_folder: str,
        idl_gtvt_folder: str,
        hyper: Dict,
    ):
        # create current patient folder
        patient_folder = os.path.join(
            idl_gtvt_folder, "patients", "patient={}".format(patient)
        )
        Folder.create(patient_folder)
        # create an empty loss.json
        Json.save(Dict(), os.path.join(patient_folder, "loss.json"))

        # copy baseline score
        baseline_score = Json.load(
            os.path.join(baseline_epoch_folder, "baseline", "score_test.json")
        )
        idl_gtvt_score_path = os.path.join(idl_gtvt_folder, "score.json")
        idl_gtvt_score = Json.load(idl_gtvt_score_path)
        for metric in g.METRICS:
            idl_gtvt_score["patient={}".format(patient)][metric][
                "round=00"
            ] = baseline_score["patient={}".format(patient)]["gtvt"][metric]
        Json.save(idl_gtvt_score, idl_gtvt_score_path)

        print("")
        print("patient:", patient)

        annotated_slices = Dict()

        # loop through each round
        max_round = max(
            len(hyper["select.step.transverse"]),
            len(hyper["select.step.coronal"]),
            len(hyper["select.step.sagittal"]),
        )
        for cur_round in range(1, max_round + 1):

            # cur round slices are add into annotated_slices in this function
            cur_round_slices = self.__select_cur_round_slices(
                annotated_slices=annotated_slices,
                hyper=hyper,
                patient_folder=patient_folder,
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
                label_folder=g.DATASET_DIR,
                hyper=hyper,
                annotated_slices=annotated_slices,
            )

            if cur_round == max_round:
                break

            # load new lr before next round
            if hyper["lr"]["reset"]:
                self.__load_next_round_lr(cur_round + 1, hyper)

        # draw avg loss of all trained patients
        self._plot_loss_fig(idl_gtvt_folder)

    def plot_loss_fig(self, idl_gtvt_id: str):
        for i in Explorer.walk_sub_folders(
            g.TRAIN_RESULTS_DIR, key_word=idl_gtvt_id
        ):
            # remove "/" if str endswith it
            if i.endswith("/"):
                i = i[:-1]
            if i.endswith(idl_gtvt_id):
                idl_gtvt_folder = i
                break
        self._plot_loss_fig(idl_gtvt_folder)

    def _plot_loss_fig(self, idl_gtvt_folder: str):
        # avg loss dict
        avg_loss = Dict()
        for cur_patient_folder in Explorer.get_sub_folders(
            os.path.join(idl_gtvt_folder, "patients"), return_full_path=True
        ):
            cur_patient_loss = Json.load(os.path.join(cur_patient_folder, "loss.json"))
            if avg_loss == {}:
                for i in cur_patient_loss:
                    avg_loss[i] = [cur_patient_loss[i]]
            else:
                for i in avg_loss:
                    avg_loss[i].append(cur_patient_loss[i])

        for i in avg_loss:
            avg_loss[i] = Value.get_avg(avg_loss[i])

        avg_loss = avg_loss.to_list()

        # draw figure
        plt.figure().clear()
        plt.plot(range(1, len(avg_loss) + 1), avg_loss, label="loss")
        plt.legend()
        plt.savefig(os.path.join(idl_gtvt_folder, "loss.png"))

    def simulation(
        self,
        baseline_id: str,
        baseline_fold: int = None,
        baseline_epoch: int = None,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        for cur_hyper in self._load_hyper_list_from_json(g.HYPER_JSON_PATH_IDL_GTVT):

            idl_gtvt_id = "idl_gtvt_" + self._init_train_id(
                train_remark=train_remark,
                debug_mode=debug_mode,
                hyper_json_path=g.HYPER_JSON_PATH_IDL_GTVT,
                hyper=cur_hyper,
            )
            print("")
            print(idl_gtvt_id)

            # find fold folder
            if baseline_fold is None or baseline_fold <= 0:
                key_word = "fold="
            else:
                key_word = "fold={:02d}".format(baseline_fold)
            baseline_fold_folder = Explorer.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_DIR, baseline_id),
                key_word=key_word,
                return_full_path=True,
            )[0]

            # find epoch folder
            if baseline_epoch is None or baseline_epoch <= 0:
                key_word = "epoch="
            else:
                key_word = "epoch={:03d}".format(baseline_epoch)
            baseline_epoch_folder = Explorer.get_sub_folders(
                baseline_fold_folder, key_word=key_word, return_full_path=True
            )[0]
            baseline_cnn_path = Explorer.get_sub_files(
                os.path.join(baseline_epoch_folder, "baseline"),
                key_word=".pt",
                return_full_path=True,
            )[0]

            # load and print hyper
            self._load_hyper(
                hyper=cur_hyper,
                baseline_cnn_path=baseline_cnn_path,
                debug_mode=debug_mode,
            )
            print("")
            self._print_hyper(cur_hyper)

            # create idl result folder
            idl_gtvt_folder = os.path.join(
                baseline_epoch_folder, "idl_gtvt", idl_gtvt_id
            )
            Folder.create(idl_gtvt_folder)

            # save hyper before training
            hyper_save_path = os.path.join(idl_gtvt_folder, "hyper.json")
            self._save_hyper(cur_hyper, hyper_save_path)

            # create an empty score json files
            Json.save(Dict(), os.path.join(idl_gtvt_folder, "score.json"))

            # training start time
            cur_hyper["time.spent.total"] = datetime.now()

            # patient loop
            for cur_patient in cur_hyper["patients"]:
                self.__training_cur_patient(
                    patient=cur_patient,
                    hyper=cur_hyper,
                    baseline_epoch_folder=baseline_epoch_folder,
                    idl_gtvt_folder=idl_gtvt_folder,
                )

                # reset cnn/optimizer/scheduler before next patient
                if cur_patient != cur_hyper["patients"][-1]:
                    self.__reset_cnn(
                        hyper=cur_hyper, baseline_cnn_path=baseline_cnn_path
                    )

            # record total time spent
            cur_hyper["time.spent.total"] = (
                datetime.now() - cur_hyper["time.spent.total"]
            )
            cur_hyper["time.spent.total"] = str(cur_hyper["time.spent.total"]).split(
                ".", 2
            )[0]

            # record avg time spent per patient
            for cur_key in cur_hyper:
                if "time.spent.avg" in cur_key:
                    cur_hyper[cur_key] /= len(cur_hyper["patients"])
                    cur_hyper[cur_key] = str(cur_hyper[cur_key]).split(".", 2)[0]

            self._save_hyper(cur_hyper, hyper_save_path)

            self.__calculate_median_score(idl_gtvt_folder)

    def calculate_median_score(self, idl_gtvt_id):
        idl_gtvt_folder = self._find_result_folder(idl_gtvt_id)
        if idl_gtvt_folder is None:
            print("idl_gtvt_id not found")
            return
        self.__calculate_median_score(idl_gtvt_folder)

    def __calculate_median_score(self, idl_gtvt_folder: str):
        score_json_path = os.path.join(idl_gtvt_folder, "score.json")
        score = Json.load(score_json_path)
        median = Dict()

        # add all patients score in to a list
        for patient in score:
            if patient == "median":
                continue
            for metric in g.METRICS:
                for cur_round in score[patient][metric]:
                    if median[metric][cur_round] == {}:
                        median[metric][cur_round] = List()
                    median[metric][cur_round].append(score[patient][metric][cur_round])

        # calculate median score
        for metric in g.METRICS:
            for cur_round in median[metric]:
                score["median"][metric][cur_round] = statistics.median(
                    median[metric][cur_round]
                )
        Json.save(data=score, path=os.path.join(score_json_path))

    def inference(self, idl_gtvt_id: str, debug_mode: bool = False):
        print("")
        print("inference: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_folder = self._find_result_folder(idl_gtvt_id)
        if idl_gtvt_folder is None:
            print("idl_gtvt_id not found")
            return

        # loop through patients folder
        patient_list = Explorer.get_sub_folders(
            os.path.join(idl_gtvt_folder, "patients"),
            key_word="patient=",
        )
        if debug_mode:
            patient_list = patient_list[:2]

        # patients with bad score
        if 0:
            patient_list = ["patient=239", "patient=260", "patient=313", "patient=180"]

        # copy baseline score
        baseline_score_json = os.path.join(
            Path(idl_gtvt_folder).parent.parent, "baseline", "score_test.json"
        )
        baseline_score = Json.load(baseline_score_json)
        idl_gtvt_score_path = os.path.join(idl_gtvt_folder, "score.json")
        if os.path.exists(idl_gtvt_score_path):
            idl_gtvt_score = Json.load(idl_gtvt_score_path)
        else:
            idl_gtvt_score = Dict()
        for cur_patient in patient_list:
            for metric in g.METRICS:
                idl_gtvt_score[cur_patient][metric]["round=00"] = baseline_score[
                    cur_patient
                ]["gtvt"][metric]
        Json.save(idl_gtvt_score, idl_gtvt_score_path)

        # loop through each patient
        for cur_patient in tqdm(patient_list):
            cur_patient_folder = os.path.join(idl_gtvt_folder, "patients", cur_patient)

            # loop through each round
            for cur_round_folder in Explorer.get_sub_folders(
                cur_patient_folder, key_word="round=", return_full_path=True
            ):
                # load current round cnn
                cur_round_cnn_path = Explorer.get_sub_files(
                    cur_round_folder, key_word=".pt", return_full_path=True
                )
                cur_round_cnn_path = cur_round_cnn_path[0]
                hyper = Dict()
                self._load_cnn(hyper=hyper, cnn_path=cur_round_cnn_path)

                self.__inference_cur_round(
                    cur_round_folder=cur_round_folder, hyper=hyper
                )

        self.__calculate_median_score(idl_gtvt_folder)
