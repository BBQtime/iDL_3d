import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from custom import GPU, Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from dataset_idl_gtvt import DataSetIDLGTVt
from loss_func_idl_gtvt import UnifiedFocalLossIDLGTVt
from numpy import ndarray
from PyQt5.QtCore import pyqtSignal
from scipy.ndimage import measurements
from str_lib import DatasetPart, DatasetVer, Metric, Plane, SelectScenario, Stat
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_core import ObsStudyProgress, TrainingCore


class ObsStudyGTVtProgress(ObsStudyProgress):
    class ProgressStep:
        INIT_CNN = 1
        INIT_DATALOADER = 2
        FIRST_BATCH = 15
        OTHER_BATCH = 1
        INFERENCE_LOAD_IMG = 3
        INFERENCE_FORWARD = 1

    def __init__(self):
        super().__init__()
        self.step = self.ProgressStep()


class TrainingIDLGTVt(TrainingCore):
    def __init__(self, idl_progress_signal: pyqtSignal = None):
        super().__init__()
        if idl_progress_signal is not None:
            self._obs_study_progress = ObsStudyGTVtProgress()
            self._obs_study_progress.progress_signal = idl_progress_signal
        else:
            self._obs_study_progress = None

    def __load_next_round_lr(self, next_round: int, hyper: Dict):
        # hyper["lr"] is a list of lr of each round
        if next_round > len(hyper["lr"]):
            next_round = len(hyper["lr"])

        if GPU.used_count() > 1:
            hyper["lr.actual"].append(hyper["lr"][next_round - 1] * GPU.used_count())
        else:
            hyper["lr.actual"].append(hyper["lr"][next_round - 1])

        self._load_hyper_optim_and_scheduler(hyper=hyper, lr=hyper["lr.actual"][-1])

    # reset cnn/optimizer/scheduler before next patient
    def __reset_cnn(self, hyper: dict, baseline_cnn_path: str):
        # reload cnn
        hyper["cnn"] = self._load_exist_cnn(baseline_cnn_path)

        self._load_hyper_optim_and_scheduler(hyper=hyper, lr=hyper["lr.actual"][0])

    def _load_hyper(
        self,
        hyper: Dict,
        baseline_id: str,
        debug_mode: bool = False,
    ):
        # load shared hyper
        super()._load_hyper(hyper)

        # iter
        if debug_mode:
            # at least 2 iters to compare loss difference
            hyper["iter"] = 1
        else:
            hyper["iter"] = Value.limit_range(hyper["iter"], (1, None))

        # lr
        # lr is saved in json file as a string, not a list, because:
        # (1) string is easier to read than list in json file (only one line)
        # (2) a "list" will be recognized as multiple trainings
        hyper["lr"] = List(hyper["lr"])
        for i in range(len(hyper["lr"])):
            hyper["lr"][i] = float(hyper["lr"][i])
            hyper["lr"][i] = Value.limit_range(hyper["lr"][i], (Value.EPS, 1))
            # check min lr, make sure it is lower than any lr in the lr list
            hyper["lr.min"] = Value.limit_range(
                hyper["lr.min"], (Value.EPS, hyper["lr"][i])
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
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            plane = "select.step.{}".format(plane)
            hyper[plane] = List(hyper[plane])
            for i in range(len(hyper[plane])):
                hyper[plane][i] = int(hyper[plane][i])
                hyper[plane][i] = Value.limit_range(hyper[plane][i], (0, None))

        # select scenario
        if hyper["select.scenario"] not in [
            SelectScenario.USER_CLICK,
            SelectScenario.LARGEST,
            SelectScenario.GRAVITY_CENTER,
            SelectScenario.EQUAL_DIVIDE,
        ]:
            hyper["select.scenario"] = SelectScenario.RANDOM

        # weight map parameters
        hyper["weight.background"] = Value.limit_range(
            hyper["weight.background"], (0.0, 1.0)
        )
        hyper["weight.selected.slice"] = Value.limit_range(
            hyper["weight.selected.slice"], (hyper["weight.background"], None)
        )
        hyper["weight.fp.fn"] = Value.limit_range(
            hyper["weight.fp.fn"], (hyper["weight.selected.slice"], None)
        )
        hyper["weight.distance.step"] = Value.limit_range(
            hyper["weight.distance.step"], (1, None)
        )
        hyper["weight.prev.round.decay"] = Value.limit_range(
            hyper["weight.prev.round.decay"], (0.0, 1.0)
        )

        # load patients, value of fold doesn't matter
        hyper["patients"] = self._load_patients(
            debug_mode=debug_mode,
            dataset_ver=hyper["dataset.ver"],
        )

        self._load_hyper_dataset_version(hyper, idl_baseline_id=baseline_id)

        # load loss function after super()._load_hyper()
        hyper["loss.func"] = UnifiedFocalLossIDLGTVt(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

        # dataset/dataloader/optimizer/scheduler will be loaded with each patient
        return

    def _simplify_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = super()._simplify_hyper(hyper)

        # here in this for loop, use "hyper" instead of "simple_hyper"
        # otherwise will cause error: dictionary changed size during iteration
        for key_name in hyper.keys():
            # "lr" and "lr.actual" are lists
            if key_name == "lr" or key_name == "lr.actual":
                simple_hyper[key_name] = hyper[key_name].to_str()

            # dont need to save "patients"
            elif key_name == "patients":
                simple_hyper.pop("patients")

            # "select.step.coronal/sagittal/transverse" are lists
            elif "select.step" in key_name:
                simple_hyper[key_name] = simple_hyper[key_name].to_str()

            # others, do nothing
            else:
                pass

        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self._simplify_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self._simplify_hyper(hyper)
        Json.save(data=simple_hyper, path=json_path)

    # in this function, cur round slices have not been added into selected_slices
    def __select_new_round_slices(
        self,
        selected_slices: Dict,
        hyper: Dict,
        patient_dir: str,
    ) -> list:  # return a list of int
        new_round_slices = Dict()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            new_round_slices[plane] = List()

        cur_round = max(
            len(selected_slices[Plane.TRANSVERSE]),
            len(selected_slices[Plane.CORONAL]),
            len(selected_slices[Plane.SAGITTAL]),
        )
        cur_round += 1

        patient = Path(patient_dir).name
        patient = patient[len("patient=") :]

        label = Nii.load(
            os.path.join(
                g.DATASET_DIR[hyper["dataset.ver"]], "HNCDL_{}_GTVt.nii".format(patient)
            ),
            binary=True,
        )
        # label_center: (d,h,w)
        label_center = measurements.center_of_mass(label)

        # select slices through each plane
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            # skip cur plane if no slice needs to be selected
            if len(hyper["select.step.{}".format(plane)]) < cur_round:
                continue

            candidates = Dict()
            ignored_slices = selected_slices[plane].to_list()

            # go through pred and record tumor size
            if plane == Plane.TRANSVERSE:
                total_slices = label.shape[0]
            elif plane == Plane.CORONAL:
                total_slices = label.shape[1]
            elif plane == Plane.SAGITTAL:
                total_slices = label.shape[2]

            for slice_num in range(total_slices):
                # skip slice that already been selected in earlier rounds
                if slice_num in ignored_slices:
                    continue
                else:
                    if plane == Plane.TRANSVERSE:
                        cur_slice_tumor_size = label[slice_num, :, :].sum()
                    elif plane == Plane.CORONAL:
                        cur_slice_tumor_size = label[:, slice_num, :].sum()
                    elif plane == Plane.SAGITTAL:
                        cur_slice_tumor_size = label[:, :, slice_num].sum()
                    # add slice with target (pred or label) into candidates
                    if cur_slice_tumor_size > 0:
                        candidates[slice_num] = cur_slice_tumor_size

            # "largest"
            if hyper["select.scenario"] == SelectScenario.LARGEST:
                # descrease sort the dict (return a list of tuple)
                candidates = candidates.sort_by_value(reverse=True)
                new_round_slices[plane] = candidates.keys()

            # "gravity.center", round = 1
            elif (
                hyper["select.scenario"] == SelectScenario.GRAVITY_CENTER
                and cur_round == 1
            ):
                if plane == Plane.TRANSVERSE:
                    new_round_slices[plane].append(round(label_center[0]))
                elif plane == Plane.CORONAL:
                    new_round_slices[plane].append(round(label_center[1]))
                elif plane == Plane.SAGITTAL:
                    new_round_slices[plane].append(round(label_center[2]))

            # "equal.divide", round = 1
            elif (
                hyper["select.scenario"] == SelectScenario.EQUAL_DIVIDE
                and cur_round == 1
            ):
                divided_parts = hyper["select.step.{}".format(plane)][0] + 1
                candidates = candidates.keys()
                for part in range(1, divided_parts):
                    idx = len(candidates) * part / divided_parts
                    idx = round(idx)
                    idx = Value.limit_range(idx, (1, len(candidates)))
                    new_round_slices[plane].append(candidates[idx - 1])

            # (1) "random"
            # (2) "gravity.center", round >= 2
            # (3) "equal.divide", round >= 2
            else:
                new_round_slices[plane] = candidates.keys()
                new_round_slices[plane].shuffle()

            # narrow new_round_slices based on select.step
            if (
                hyper["select.scenario"] == SelectScenario.GRAVITY_CENTER
                and cur_round == 1
            ):
                new_slices_num = 1
            else:
                new_slices_num = hyper["select.step.{}".format(plane)][cur_round - 1]
            if new_slices_num < len(new_round_slices[plane]):
                new_round_slices[plane] = new_round_slices[plane][:new_slices_num]

            # add new_round_slices into selected_slices
            selected_slices[plane]["round={:02d}".format(cur_round)] = new_round_slices[
                plane
            ]

        return new_round_slices

    def __get_label_masked_by_selected_slices(self, round_dir: str, dataset_ver: str):
        cur_round = Path(round_dir).name
        patient_dir = Path(round_dir).parent
        patient = patient_dir.name
        # change "patient=123" into "123"
        patient = patient[len("patient=") :]

        label = Nii.load(
            os.path.join(
                g.DATASET_DIR[dataset_ver], "HNCDL_{}_GTVt.nii".format(patient)
            ),
            binary=True,
        )

        selected_slices = Json.load(os.path.join(patient_dir, "selected_slices.json"))

        # selected slices mask
        selected_slices_mask = Dict()

        # loop through each plane
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            selected_slices_mask[plane] = np.zeros(label.shape, dtype=np.float32)

            # loop through each round
            for cur_round in selected_slices[plane]:
                # str to list
                selected_slices[plane][cur_round] = List(
                    selected_slices[plane][cur_round]
                )
                # current step
                for slice_num in selected_slices[plane][cur_round]:
                    # change slice id from str into int
                    slice_num = int(slice_num)
                    if plane == Plane.TRANSVERSE:
                        selected_slices_mask[plane][slice_num, :, :] = np.ones_like(
                            selected_slices_mask[plane][0, :, :]
                        )
                    elif plane == Plane.CORONAL:
                        selected_slices_mask[plane][:, slice_num, :] = np.ones_like(
                            selected_slices_mask[plane][:, 0, :]
                        )
                    elif plane == Plane.SAGITTAL:
                        selected_slices_mask[plane][:, :, slice_num] = np.ones_like(
                            selected_slices_mask[plane][:, :, 0]
                        )

        # combine selected_slices_mask on 3 anatomical planes
        selected_slices_mask = np.maximum(
            np.maximum(
                selected_slices_mask[Plane.TRANSVERSE],
                selected_slices_mask[Plane.CORONAL],
            ),
            selected_slices_mask[Plane.SAGITTAL],
        )

        return label * selected_slices_mask

    def __inference_cur_round(
        self,
        round_dir: str,
        cnn,
        dataset_ver: str,
        no_pt: str,
        segment_metrics: Dict = None,
    ):
        cur_round = Path(round_dir).name

        patient = Path(round_dir).parent.name

        # get "selected sliced masked label" for post processing
        idl_gtvt_label_masked_by_selected_slices = (
            self.__get_label_masked_by_selected_slices(
                round_dir=round_dir, dataset_ver=dataset_ver
            )
        )

        # result structure: gtvt: {pred, dsc, msd, hd95}
        patient_outputs = self._inference_single_patient(
            patient=patient[len("patient=") :],
            cnn=cnn,
            dataset_ver=dataset_ver,
            dataset_part=DatasetPart.TEST,
            no_pt=no_pt,
            segment_metrics=segment_metrics,
            idl_gtvt_label_masked_by_selected_slices=idl_gtvt_label_masked_by_selected_slices,
        )

        # save score of cur patient
        if self._obs_study_progress is None:
            idl_gtvt_dir = Path(round_dir).parent.parent.parent
            score_json_path = os.path.join(
                idl_gtvt_dir, "inference_{}.json".format(dataset_ver)
            )
            if os.path.exists(score_json_path):
                score = Json.load(score_json_path)
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    score[patient][metric][cur_round] = patient_outputs["gtvt"][metric]
                Json.save(score, score_json_path)

        # save pred of cur patient
        Nii.save(
            img=patient_outputs["gtvt"]["pred"],
            save_path=os.path.join(round_dir, "gtvt_pred.nii.gz"),
            spacing=g.NII_SPACING,
        )

    def __training_single_round(
        self,
        round_dir: str,
        hyper: Dict,
        selected_slices: Dict,
        segment_metrics: Dict = None,
    ):
        Dir.create(round_dir)

        cur_round = Path(round_dir).name
        cur_round = int(cur_round[len("round=") :])

        patient_dir = Path(round_dir).parent
        patient = patient_dir.name[len("patient=") :]

        idl_gtvt_dir = patient_dir.parent.parent
        loss_json_path = os.path.join(patient_dir, "loss.json")
        loss_dict = Json.load(loss_json_path)

        # pred dir
        if cur_round == 1:
            pred_dir = os.path.join(
                idl_gtvt_dir.parent,
                "baseline",
                "patients",
                "patient={}".format(patient),
            )
        else:
            pred_dir = os.path.join(patient_dir, "round={:02d}".format(cur_round - 1))

        # delineation dir
        if hyper["select.scenario"] == SelectScenario.USER_CLICK:
            delineation_dir = os.path.join(patient_dir, "round=01")
        else:
            delineation_dir = None

        # record current round time spent
        time_spent = datetime.now()

        # create iDL dataset
        augment = Dict()
        augment["augment.methods"] = hyper["augment.methods"]
        augment["augment.pct"] = hyper["augment.pct"]
        augment["augment.times"] = hyper["augment.times"]
        augment["augment.min"] = hyper["augment.min"]
        augment["augment.max"] = hyper["augment.max"]

        weight = Dict()
        weight["weight.background"] = hyper["weight.background"]
        weight["weight.distance.step"] = hyper["weight.distance.step"]
        weight["weight.fp.fn"] = hyper["weight.fp.fn"]
        weight["weight.prev.round.decay"] = hyper["weight.prev.round.decay"]
        weight["weight.selected.slice"] = hyper["weight.selected.slice"]

        dataset_idl_gtvt = DataSetIDLGTVt(
            patient=patient,
            selected_slices=selected_slices,
            pred_dir=pred_dir,
            delineation_dir=delineation_dir,  # delineation_dir=None for simulation
            dataset_ver=hyper["dataset.ver"],
            no_pt=hyper["no.pt"],
            augment=augment,
            weight=weight,
        )

        # optimize batch size (before create dataloader)
        self._optimize_batch_size(dataset=dataset_idl_gtvt, hyper=hyper)

        # idl gtvt dataloader
        idl_gtvt_loader = DataLoader(
            dataset=dataset_idl_gtvt,
            batch_size=hyper["batch.size.actual"],
            shuffle=True,
            num_workers=g.NUM_WORKERS,
        )

        # idl progress INIT_DATALOADER
        # self._timer.cal_duration("INIT_DATALOADER")
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step += (
                self._obs_study_progress.step.INIT_DATALOADER
            )
            self._obs_study_progress.emit_signal()

        # function is runing in QThread, run in background and hide tqdm bar
        if self._obs_study_progress is not None:
            iter_range = range(hyper["iter"])

        # not in QThread, show tqdm bar
        else:
            iter_range = tqdm(range(hyper["iter"]))

        # iter loop through iterations
        for cur_iter in iter_range:
            hyper["cnn"].train()
            iter_loss = 0
            batch_count = 0

            # freeze layers before iDL
            if hyper["layer.freezing"]:
                if GPU.used_count() > 1:
                    # here, hyper["cnn"] is DataParallel, not network itself
                    hyper["cnn"].module.freeze_top()
                else:
                    hyper["cnn"].freeze_top()

            # here, labels only have 2 channels: background and gtvt, No gtvn
            for input_imgs, labels, weight_map in idl_gtvt_loader:
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                input_imgs = input_imgs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                weight_map = weight_map.to(g.DEVICE)
                preds = hyper["cnn"](input_imgs)
                loss = hyper["loss.func"](preds, labels, weight_map)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                iter_loss += loss.item()
                batch_count += 1

                # idl progress MINI_BATCH
                if batch_count <= 1:
                    # self._timer.cal_duration("FIRST_BATCH")
                    if self._obs_study_progress is not None:
                        self._obs_study_progress.cur_step += (
                            self._obs_study_progress.step.FIRST_BATCH
                        )
                        self._obs_study_progress.emit_signal()
                else:
                    # self._timer.cal_duration("OTHER_BATCH")
                    if self._obs_study_progress is not None:
                        self._obs_study_progress.cur_step += (
                            self._obs_study_progress.step.OTHER_BATCH
                        )
                        self._obs_study_progress.emit_signal()

            # cur iter finished
            # update scheduler
            iter_loss /= batch_count
            hyper["scheduler"].step(iter_loss)

            # record loss
            loss_dict[
                "iter={:03d}".format((cur_round - 1) * hyper["iter"] + (cur_iter + 1))
            ] = iter_loss
            # save loss and update loss figure after every iter, if there is only one patient
            patient_dirs_list = Dir.get_sub_dirs(os.path.join(idl_gtvt_dir, "patients"))
            if len(patient_dirs_list) <= 1:
                Json.save(loss_dict, loss_json_path)
                self._plot_loss_fig(idl_gtvt_dir)

        # current round idl finished
        # save cnn
        cnn_save_path = os.path.join(round_dir, Path(round_dir).name + ".pt")
        self._save_cnn(hyper, cnn_save_path)

        # save selected_slices dict before inference, because masked_label needs it
        # copy a new dict to avoid changing origin selected_slices dict
        # dont need to save it for observer study
        if hyper["select.scenario"] != SelectScenario.USER_CLICK:
            selected_slices_to_save = selected_slices.copy()
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                for cur_round in selected_slices_to_save[plane]:
                    selected_slices_to_save[plane][cur_round] = selected_slices_to_save[
                        plane
                    ][cur_round].to_str()
            Json.save(
                data=selected_slices_to_save,
                path=os.path.join(patient_dir, "selected_slices.json"),
            )

        # inference
        self.__inference_cur_round(
            round_dir=round_dir,
            cnn=hyper["cnn"],
            dataset_ver=hyper["dataset.ver"],
            no_pt=hyper["no.pt"],
            segment_metrics=segment_metrics,
        )

        # save time spent
        time_spent = datetime.now() - time_spent
        # key_name like: "time.spent.round=01"
        key_name = "time.spent." + Path(round_dir).name
        if hyper[key_name] == {}:
            hyper[key_name] = time_spent
        else:
            hyper[key_name] += time_spent

        # save loss
        Json.save(loss_dict, loss_json_path)

    def __simulation_single_patient(
        self,
        patient: str,
        idl_gtvt_dir: str,
        hyper: Dict,
        segment_metrics: Dict = None,
    ):
        print("")
        print("patient:", patient)

        # create current patient folder
        patient_dir = os.path.join(
            idl_gtvt_dir, "patients", "patient={}".format(patient)
        )
        Dir.create(patient_dir)
        # create an empty loss.json
        Json.save(Dict(), os.path.join(patient_dir, "loss.json"))

        # copy baseline scores
        baseline_score = Json.load(
            os.path.join(
                Path(idl_gtvt_dir).parent,
                "baseline",
                "inference_{}.json".format(hyper["dataset.ver"]),
            )
        )
        idl_gtvt_score = Json.load(
            os.path.join(
                idl_gtvt_dir,
                "inference_{}.json".format(hyper["dataset.ver"]),
            )
        )
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            idl_gtvt_score["patient={}".format(patient)][metric]["round=00"] = (
                baseline_score["patient={}".format(patient)]["gtvt"][metric]
            )
        Json.save(
            idl_gtvt_score,
            os.path.join(
                idl_gtvt_dir,
                "inference_{}.json".format(hyper["dataset.ver"]),
            ),
        )

        selected_slices = Dict()
        max_round = max(
            len(hyper["select.step.transverse"]),
            len(hyper["select.step.coronal"]),
            len(hyper["select.step.sagittal"]),
        )
        # loop through each round
        for cur_round in range(1, max_round + 1):
            # new_round_slices are add into selected_slices in this function
            new_round_slices = self.__select_new_round_slices(
                selected_slices=selected_slices,
                hyper=hyper,
                patient_dir=patient_dir,
            )
            # no slice needs to be selected in cur round
            if (
                len(new_round_slices[Plane.TRANSVERSE]) == 0
                and len(new_round_slices[Plane.CORONAL]) == 0
                and len(new_round_slices[Plane.SAGITTAL]) == 0
            ):
                break

            # start current round
            print("round:", cur_round)

            round_dir = os.path.join(patient_dir, "round={:02d}".format(cur_round))

            self.__training_single_round(
                round_dir=round_dir,
                hyper=hyper,
                selected_slices=selected_slices,
                segment_metrics=segment_metrics,
            )

            if cur_round == max_round:
                break

            # load new lr before next round
            if hyper["lr"]["reset"]:
                self.__load_next_round_lr(cur_round + 1, hyper)

        # draw avg loss of all trained patients
        self._plot_loss_fig(idl_gtvt_dir)

    def plot_loss_fig(self, idl_gtvt_id: str):
        for i in Dir.walk_sub_dirs(g.TRAIN_RESULTS_DIR, key_word=idl_gtvt_id):
            # remove "/" if str endswith it
            if i.endswith("/"):
                i = i[:-1]
            if i.endswith(idl_gtvt_id):
                idl_gtvt_dir = i
                break
        self._plot_loss_fig(idl_gtvt_dir)

    def _plot_loss_fig(self, idl_gtvt_dir: str):
        # avg loss dict
        avg_loss = Dict()
        for patient_dir in Dir.get_sub_dirs(
            os.path.join(idl_gtvt_dir, "patients"), full_path=True
        ):
            loss_path = os.path.join(patient_dir, "loss.json")
            if os.path.exists(loss_path):
                cur_patient_loss = Json.load(loss_path)
                if avg_loss == {}:
                    for i in cur_patient_loss:
                        avg_loss[i] = [cur_patient_loss[i]]
                else:
                    for i in avg_loss:
                        avg_loss[i].append(cur_patient_loss[i])
            else:
                continue

        if len(avg_loss) > 0:
            for i in avg_loss:
                avg_loss[i] = Value.avg(avg_loss[i])

            avg_loss = avg_loss.to_list()

            # draw figure
            plt.figure().clear()
            plt.plot(range(1, len(avg_loss) + 1), avg_loss, label="loss")
            plt.legend()
            plt.savefig(os.path.join(idl_gtvt_dir, "loss.png"))

    def __find_best_baseline_fold_cnn(self, baseline_id: str) -> str:
        scores = Dict()

        fold_dirs = Dir.get_sub_dirs(
            input_dir=os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, "baseline"),
            key_word="fold=",
            full_path=True,
        )
        for fold_dir in fold_dirs:
            baseline_dataset_ver = Json.load(os.path.join(fold_dir, "hyper.json"))[
                "dataset.ver"
            ]

            fold = Path(fold_dir).name
            epoch_dir = Dir.get_sub_dirs(fold_dir, key_word="epoch=", full_path=True)[0]
            epoch_scores = Json.load(
                os.path.join(
                    epoch_dir,
                    "inference_{}.json".format(baseline_dataset_ver),
                )
            )
            for stat in [Stat.MEDIAN, Stat.AVG]:
                scores[fold][stat] = epoch_scores[stat]

        for stat in [Stat.MEDIAN, Stat.AVG]:
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    # create a tmp list to sort
                    list_to_sort = List()
                    # add elements into the list
                    for epoch in scores.keys():
                        list_to_sort.append(scores[epoch][stat][gtv][metric])
                    # sort the list
                    if metric == Metric.DSC:
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)
                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        new_value = list_to_sort.index(scores[epoch][stat][gtv][metric])
                        # if metric == Metric.DSC:
                        #     new_value *= 2
                        scores[epoch][stat][gtv][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stat in [Stat.AVG, Stat.MEDIAN]:
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                        evaluation[epoch] += scores[epoch][stat][gtv][metric]

        best_fold = evaluation.key_with_max_value()
        best_fold_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, baseline_id, "baseline", best_fold
        )
        best_epoch_dir = Dir.get_sub_dirs(
            best_fold_dir, key_word="epoch=", full_path=True
        )[0]
        best_cnn_path = Dir.get_sub_files(
            best_epoch_dir, key_word=".pt", full_path=True
        )[0]
        return best_cnn_path

    def simulation(
        self,
        baseline_id: str,
        dataset_ver: str = None,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        # load baseline data
        self._is_valid_baseline_id(baseline_id)
        baseline_dir = self._find_train_dir(baseline_id)
        if baseline_dir is None:
            Debug.error_exit("Can not find 'baseline_id'!")

        baseline_fold_dirs = Dir.get_sub_dirs(
            baseline_dir, key_word="fold=", full_path=True
        )
        baseline_hyper = Json.load(os.path.join(baseline_fold_dirs[0], "hyper.json"))
        no_pt = baseline_hyper["no.pt"]
        baseline_dataset_ver = baseline_hyper["dataset.ver"]

        # check dataset version and dataset part
        dataset_ver = self._is_valid_dataset_version(
            dataset_ver=dataset_ver,
            origin_dataset_ver=baseline_dataset_ver,
        )

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics()

        # load hyper
        hyper_series = self._load_hyper_series_from_json(g.HYPER_JSON_PATH["idl.gtvt"])

        for hyper in hyper_series:
            hyper["dataset.ver"] = dataset_ver

            # idl.gtvt doesnt have "no.pt" hyperparam, copy it from baseline
            hyper["no.pt"] = no_pt

            idl_gtvt_id = "idl.gtvt_" + self._init_train_id(
                hyper=hyper,
                hyper_json_path=g.HYPER_JSON_PATH["idl.gtvt"],
                train_remark=train_remark,
                debug_mode=debug_mode,
            )

            print("")
            print(idl_gtvt_id)

            # create idl result dir
            idl_gtvt_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvt_id)
            Dir.create(idl_gtvt_dir)

            # load and print hyper
            self._load_hyper(
                hyper=hyper,
                baseline_id=baseline_id,
                debug_mode=debug_mode,
            )
            print("")
            self._print_hyper(hyper)

            # save hyper before training
            hyper_save_path = os.path.join(idl_gtvt_dir, "hyper.json")
            self._save_hyper(hyper, hyper_save_path)

            # training start time
            hyper["time.spent.total"] = datetime.now()

            # create an empty score json files
            Json.save(
                Dict(),
                os.path.join(
                    idl_gtvt_dir,
                    "inference_{}.json".format(hyper["dataset.ver"]),
                ),
            )

            # best baseline cnn is decided by dataset_part
            baseline_cnn_path = self.__find_best_baseline_fold_cnn(baseline_id)

            patient_list = hyper["patients"][DatasetPart.TEST]

            # loop through each patient
            for patient in patient_list:
                self.__reset_cnn(
                    hyper=hyper,
                    baseline_cnn_path=baseline_cnn_path,
                )

                self.__simulation_single_patient(
                    patient=patient,
                    idl_gtvt_dir=idl_gtvt_dir,
                    hyper=hyper,
                    segment_metrics=segment_metrics,
                )

                # calculate and save avg and median scores
                self._inference_calculate_save_avg_median(
                    idl_gtvt_dir,
                    dataset_ver=hyper["dataset.ver"],
                )

            # record total time spent
            hyper["time.spent.total"] = datetime.now() - hyper["time.spent.total"]
            hyper["time.spent.total"] = str(hyper["time.spent.total"]).split(".", 2)[0]

            # record avg time spent per patient
            for key_name in hyper.keys():
                if "time.spent.round" in key_name:
                    # devided by number of all test patients
                    hyper[key_name] /= len(hyper["patients"].to_list())
                    hyper[key_name] = str(hyper[key_name]).split(".", 2)[0]

            self._save_hyper(hyper, hyper_save_path)

    def obs_study(
        self,
        idl_gtvt_id: str,
        patient: str,
        debug_mode: bool = False,
    ):
        print("")
        print("observer study: {}".format(idl_gtvt_id))

        # load baseline data
        baseline_id = "baseline_obs.study"
        baseline_dir = self._find_train_dir(baseline_id)

        baseline_fold_dirs = Dir.get_sub_dirs(
            baseline_dir, key_word="fold=", full_path=True
        )
        baseline_hyper = Json.load(os.path.join(baseline_fold_dirs[0], "hyper.json"))
        no_pt = baseline_hyper["no.pt"]

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics()

        # create idl result dir
        idl_gtvt_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvt_id)
        Dir.create(idl_gtvt_dir)

        # create current patient folder
        patient_dir = os.path.join(
            idl_gtvt_dir, "patients", "patient={}".format(patient)
        )
        Dir.create(patient_dir)

        # load hyper
        hyper_series = self._load_hyper_series_from_json(g.HYPER_JSON_PATH["idl.gtvt"])
        hyper = hyper_series[0]

        # idl.gtvt doesnt have "no.pt" hyperparam, copy it from baseline
        hyper["no.pt"] = no_pt

        # change dataset version to OBS_STUDY
        hyper["dataset.ver"] = DatasetVer.OBS_STUDY

        # select scenario
        selected_slices_path = os.path.join(patient_dir, "selected_slices.json")
        if os.path.exists(selected_slices_path):
            hyper["select.scenario"] = SelectScenario.USER_CLICK
        else:  # if not running in qthread, simulate user click using gravity center
            hyper["select.scenario"] = SelectScenario.GRAVITY_CENTER

        hyper["select.step.transverse"] = "1"
        hyper["select.step.coronal"] = "1"
        hyper["select.step.sagittal"] = "1"

        # load and print hyper
        self._load_hyper(
            hyper=hyper,
            baseline_id=baseline_id,
            debug_mode=debug_mode,
        )

        # save hyper before training
        hyper_save_path = os.path.join(idl_gtvt_dir, "hyper.json")
        self._save_hyper(hyper, hyper_save_path)

        # training start time
        hyper["time.spent.total"] = datetime.now()

        # idl progress init (after load hyper)
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step = 0
            dataset_len = hyper["augment.times"]
            dataset_len /= hyper["batch.size.actual"]
            dataset_len = math.ceil(dataset_len)
            self._obs_study_progress.total_step = (
                self._obs_study_progress.step.INIT_CNN
                + self._obs_study_progress.step.INIT_DATALOADER
                + self._obs_study_progress.step.INFERENCE_LOAD_IMG
                + self._obs_study_progress.step.INFERENCE_FORWARD
            )
            self._obs_study_progress.total_step += hyper["iter"] * (
                self._obs_study_progress.step.FIRST_BATCH
                + self._obs_study_progress.step.OTHER_BATCH * (dataset_len - 1)
            )

        # best baseline cnn is decided by dataset_part
        baseline_cnn_path = self.__find_best_baseline_fold_cnn(baseline_id)

        self.__reset_cnn(
            hyper=hyper,
            baseline_cnn_path=baseline_cnn_path,
        )

        # idl progress INIT_CNN
        # self._timer.cal_duration("INIT_CNN")
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step += self._obs_study_progress.step.INIT_CNN
            self._obs_study_progress.emit_signal()

        # create an empty loss.json
        Json.save(Dict(), os.path.join(patient_dir, "loss.json"))

        # load selected slices
        if os.path.exists(selected_slices_path):
            selected_slices = Json.load(
                os.path.join(patient_dir, "selected_slices.json"),
            )
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                selected_slices[plane]["round=01"] = List(
                    int(selected_slices[plane]["round=01"])
                )
        else:
            selected_slices = Dict()
            self.__select_new_round_slices(
                selected_slices=selected_slices,
                hyper=hyper,
                patient_dir=patient_dir,
            )

        self.__training_single_round(
            round_dir=os.path.join(patient_dir, "round=01"),
            hyper=hyper,
            selected_slices=selected_slices,
            segment_metrics=segment_metrics,
        )

        # draw avg loss of all trained patients
        self._plot_loss_fig(idl_gtvt_dir)

        # record total time spent
        hyper["time.spent.total"] = datetime.now() - hyper["time.spent.total"]
        hyper["time.spent.total"] = str(hyper["time.spent.total"]).split(".", 2)[0]

        # record avg time spent per patient
        for key_name in hyper.keys():
            if "time.spent.round" in key_name:
                # devided by number of all test patients
                hyper[key_name] /= len(hyper["patients"].to_list())
                hyper[key_name] = str(hyper[key_name]).split(".", 2)[0]

        self._save_hyper(hyper, hyper_save_path)

        # if self._obs_study_progress is not None:
        #     print(self._obs_study_progress.cur_step, self._obs_study_progress.total_step)

    def inference_calculate_save_avg_median(self, idl_gtvt_id: str):
        print("")
        print("calculate and save avg and median score: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_dir = self._find_train_dir(idl_gtvt_id)
        if idl_gtvt_dir is None:
            Debug.error_exit("'idl_gtvt_id' not found!")

        hyper = Json.load(os.path.join(idl_gtvt_dir, "hyper.json"))
        dataset_ver = hyper["dataset.ver"]
        dataset_ver = self._is_valid_dataset_version(dataset_ver=dataset_ver)
        print("dataset version: {}".format(dataset_ver))

        self._inference_calculate_save_avg_median(
            idl_gtvt_dir=idl_gtvt_dir,
            dataset_ver=dataset_ver,
        )

    def _inference_calculate_save_avg_median(
        self,
        idl_gtvt_dir: str,
        dataset_ver: str,
    ):
        score_json_path = os.path.join(
            idl_gtvt_dir, "inference_{}.json".format(dataset_ver)
        )
        scores = Json.load(score_json_path)
        all_patient_scores = Dict()

        # add all patients score in to a list
        for patient in scores:
            if patient == Stat.MEDIAN or patient == Stat.AVG:
                continue
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                for cur_round in scores[patient][metric]:
                    if all_patient_scores[metric][cur_round] == {}:
                        all_patient_scores[metric][cur_round] = List()
                    all_patient_scores[metric][cur_round].append(
                        scores[patient][metric][cur_round]
                    )
        # calculate median score
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            for cur_round in all_patient_scores[metric]:
                scores[Stat.MEDIAN][metric][cur_round] = Value.median(
                    all_patient_scores[metric][cur_round]
                )
                scores[Stat.AVG][metric][cur_round] = Value.avg(
                    all_patient_scores[metric][cur_round]
                )
        Json.save(data=scores, path=os.path.join(score_json_path))

    def inference(self, idl_gtvt_id: str, debug_mode: bool = False):
        print("")
        print("inference: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_dir = self._find_train_dir(idl_gtvt_id)
        if idl_gtvt_dir is None:
            Debug.error_exit("'idl_gtvt_id' not found!")

        # load dataset version
        hyper = Json.load(os.path.join(idl_gtvt_dir, "hyper.json"))
        dataset_ver = hyper["dataset.ver"]
        no_pt = hyper["no.pt"]
        dataset_ver = self._is_valid_dataset_version(dataset_ver=dataset_ver)
        print("dataset version: {}".format(dataset_ver))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics()

        # get all patients
        patients = self._load_patients(
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )
        patients = patients[DatasetPart.TEST]

        # copy baseline score
        baseline_score = Json.load(
            os.path.join(
                Path(idl_gtvt_dir).parent,
                "baseline",
                "inference_{}.json".format(dataset_ver),
            )
        )
        idl_gtvt_score_path = os.path.join(
            idl_gtvt_dir, "inference_{}.json".format(dataset_ver)
        )
        if os.path.exists(idl_gtvt_score_path):
            idl_gtvt_score = Json.load(idl_gtvt_score_path)
        else:
            idl_gtvt_score = Dict()
        for patient in patients:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                idl_gtvt_score["patient={}".format(patient)][metric]["round=00"] = (
                    baseline_score["patient={}".format(patient)]["gtvt"][metric]
                )
        Json.save(idl_gtvt_score, idl_gtvt_score_path)

        # loop through each patient
        for patient in tqdm(patients):
            patient_dir = os.path.join(
                idl_gtvt_dir, "patients", "patient={}".format(patient)
            )

            # loop through each round
            for round_dir in Dir.get_sub_dirs(
                patient_dir, key_word="round=", full_path=True
            ):
                # load current round cnn
                cnn_path = Dir.get_sub_files(round_dir, key_word=".pt", full_path=True)[
                    0
                ]
                cnn = self._load_exist_cnn(cnn_path)

                self.__inference_cur_round(
                    round_dir=round_dir,
                    cnn=cnn,
                    dataset_ver=dataset_ver,
                    no_pt=no_pt,
                    segment_metrics=segment_metrics,
                )

        self._inference_calculate_save_avg_median(
            idl_gtvt_dir=idl_gtvt_dir,
            dataset_ver=dataset_ver,
        )

    def _inference_single_patient_record_labels(self, labels: Dict):
        outputs = Dict()
        outputs["gtvt"]["label"] = labels["gtvt"]
        return outputs

    def _inference_single_patient_record_outputs(
        self, outputs: Dict, preds: Dict, input_imgs: Tensor, idl_gtvn_clicks: Tensor
    ):
        outputs["gtvt"]["pred"] = preds[1]

    def _inference_single_patient_gtvt_post_process(
        self, outputs: Dict, idl_gtvt_label_masked_by_selected_slices: ndarray
    ):
        if idl_gtvt_label_masked_by_selected_slices is not None:
            cc_list = Img.connected_components(outputs["gtvt"]["pred"])
            outputs["gtvt"]["pred"] = np.zeros_like(outputs["gtvt"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * idl_gtvt_label_masked_by_selected_slices).sum() > 0:
                    outputs["gtvt"]["pred"] = np.maximum(
                        outputs["gtvt"]["pred"], cur_cc
                    )

    def _is_valid_dataset_part(self, dataset_part: str):
        if dataset_part != DatasetPart.TEST:
            Debug.error_exit("'dataset_part' must be 'TEST'!")
