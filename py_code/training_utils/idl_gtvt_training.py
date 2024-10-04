import os
import random
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path

import global_utils.global_core as g
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dataset_utils.idl_gtvt_dataset import IDLGTVtDataSet
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import (
    DatasetPart,
    DatasetVer,
    ErrMsg,
    Metric,
    Plane,
    SelectScenario,
    Stat,
)
from loss_utils.idl_gtvt_loss import IDLGTVtLoss
from numpy import ndarray

# from PyQt5.QtCore import pyqtSignal
from scipy.ndimage import measurements
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_utils.training_core import ObsStudyProgress, TrainingCore

matplotlib.use("Agg")


class GTVtObsStudyProgress(ObsStudyProgress):
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


class IDLGTVtTraining(TrainingCore):
    def __init__(self):  # , idl_progress_signal: None):
        super().__init__()
        # if idl_progress_signal is not None:
        #     self._obs_study_progress = GTVtObsStudyProgress()
        #     self._obs_study_progress.progress_signal = idl_progress_signal
        # else:
        #     self._obs_study_progress = None

    def __load_next_round_lr(self, next_round: int, hyper: Dict):
        # hyper["lr"] is a list of lr of each round
        if next_round > len(hyper["lr"]):
            next_round = len(hyper["lr"])

        if g.used_gpu_count() > 1:
            hyper["lr.actual"].append(hyper["lr"][next_round - 1] * g.used_gpu_count())
        else:
            hyper["lr.actual"].append(hyper["lr"][next_round - 1])

        self._load_hyper_optim_and_scheduler(hyper=hyper, lr=hyper["lr.actual"][-1])

    # reset cnn/optimizer/scheduler before next patient
    def __reset_cnn(self, hyper: dict, baseline_cnn_path: str, device_id: int = 0):
        # reload cnn
        hyper["cnn"] = self._load_exist_cnn(baseline_cnn_path, device_id=device_id)

        self._load_hyper_optim_and_scheduler(hyper=hyper, lr=hyper["lr.actual"][0])

    def _load_hyper(
        self,
        hyper: Dict,
        baseline_id: str,
        debug_mode: bool = False,
        device_id: int = 0,
    ):
        # load shared hyper
        super()._load_hyper(hyper)

        # iter
        if debug_mode:
            # at least 2 iters to compare loss difference
            hyper["iter"] = 1
        else:
            hyper["iter"] = g.clamp_value(hyper["iter"], (1, None))

        # lr
        # lr is saved in json file as a string, not a list, because:
        # (1) string is easier to read than list in json file (only one line)
        # (2) a "list" will be recognized as multiple trainings
        hyper["lr"] = List(hyper["lr"])
        for i in range(len(hyper["lr"])):
            hyper["lr"][i] = float(hyper["lr"][i])
            hyper["lr"][i] = g.clamp_value(hyper["lr"][i], (g.EPS, 1))
            # check min lr, make sure it is lower than any lr in the lr list
            hyper["lr.min"] = g.clamp_value(hyper["lr.min"], (g.EPS, hyper["lr"][i]))

        # actual lr
        hyper["lr.actual"] = List()
        if g.used_gpu_count() > 1:
            hyper["lr.actual"].append(hyper["lr"][0] * g.used_gpu_count())
        else:
            hyper["lr.actual"].append(hyper["lr"][0])

        # lr decay patience (before shared hyper)
        hyper["lr.decay.patience"] = g.clamp_value(
            hyper["lr.decay.patience"], (1, hyper["iter"])
        )

        # augmentation times
        hyper["augment.times"] = g.clamp_value(hyper["augment.times"], (1, None))

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
                hyper[plane][i] = g.clamp_value(hyper[plane][i], (0, None))

        # select scenario
        if hyper["select.scenario"] not in [
            SelectScenario.USER_CLICK,
            SelectScenario.LARGEST,
            SelectScenario.GRAVITY_CENTER,
            SelectScenario.BIAS_GRAVITY_CENTER,
            SelectScenario.EQUAL_DIVIDE,
        ]:
            hyper["select.scenario"] = SelectScenario.RANDOM

        # weight map parameters
        hyper["weight.background"] = g.clamp_value(
            hyper["weight.background"], (0.0, 1.0)
        )
        hyper["weight.selected.slice"] = g.clamp_value(
            hyper["weight.selected.slice"], (hyper["weight.background"], None)
        )
        hyper["weight.fp.fn"] = g.clamp_value(
            hyper["weight.fp.fn"], (hyper["weight.selected.slice"], None)
        )
        hyper["weight.distance.step"] = g.clamp_value(
            hyper["weight.distance.step"], (1, None)
        )
        hyper["weight.prev.round.decay"] = g.clamp_value(
            hyper["weight.prev.round.decay"], (0.0, 1.0)
        )

        # load dataset version
        self._load_hyper_dataset_version(
            hyper=hyper,
            idl_baseline_id=baseline_id,
        )

        # load patients, value of fold doesn't matter
        hyper["patients"] = self._load_patients(
            debug_mode=debug_mode,
            dataset_ver=hyper["dataset.ver"],
        )

        # load loss function after super()._load_hyper()
        hyper["loss.func"] = IDLGTVtLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICES[device_id])

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
        g.save_json(data=simple_hyper, path=json_path)

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

        label = g.load_gtv_labels(
            dataset_ver=hyper["dataset.ver"],
            patient=patient,
        )["gtvt"]
        if label.max() == 0:
            g.error_exit("label is empty!")

        # get gravity center of label
        if (
            hyper["select.scenario"]
            in [SelectScenario.GRAVITY_CENTER, SelectScenario.BIAS_GRAVITY_CENTER]
            and cur_round == 1
        ):
            d, h, w = measurements.center_of_mass(label)
            # float to int
            gravity_center = (round(d), round(h), round(w))

            if hyper["select.scenario"] == SelectScenario.GRAVITY_CENTER:
                gtvt_click = gravity_center

            # simulate biased gravity center
            elif hyper["select.scenario"] == SelectScenario.BIAS_GRAVITY_CENTER:

                while_counter = 0
                while 1:
                    while_counter += 1
                    # cant find biased gravity center after 50 times attempts
                    # use gravity center instead
                    if while_counter >= 50:
                        print("bias gravity center - while counter: ", while_counter)
                        gtvt_click = gravity_center
                        break

                    # add random bias
                    random_bias = (
                        random.randint(-5, 5),
                        random.randint(-5, 5),
                        random.randint(-5, 5),
                    )
                    d, h, w = gravity_center
                    d += random_bias[0]
                    h += random_bias[1]
                    w += random_bias[2]

                    # check if biased label center is inside label
                    if (
                        0 <= d < label.shape[0]
                        and 0 <= h < label.shape[1]
                        and 0 <= w < label.shape[2]
                        and label[d, h, w] > 0
                    ):
                        print("bias gravity center - while counter: ", while_counter)
                        gtvt_click = (d, h, w)
                        break
                    else:
                        continue

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
                hyper["select.scenario"]
                in [SelectScenario.GRAVITY_CENTER, SelectScenario.BIAS_GRAVITY_CENTER]
                and cur_round == 1
            ):
                d, h, w = gtvt_click
                if plane == Plane.TRANSVERSE:
                    new_round_slices[plane].append(d)
                elif plane == Plane.CORONAL:
                    new_round_slices[plane].append(h)
                elif plane == Plane.SAGITTAL:
                    new_round_slices[plane].append(w)

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
                    idx = g.clamp_value(idx, (1, len(candidates)))
                    new_round_slices[plane].append(candidates[idx - 1])

            # (1) "random"
            # (2) "gravity.center", round >= 2
            # (3) "equal.divide", round >= 2
            else:
                new_round_slices[plane] = candidates.keys()
                new_round_slices[plane].shuffle()

            # make sure number of new_round_slices is no more than select.step
            if (
                hyper["select.scenario"]
                in [SelectScenario.GRAVITY_CENTER, SelectScenario.BIAS_GRAVITY_CENTER]
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

        label = g.load_gtv_labels(
            dataset_ver=dataset_ver,
            patient=patient,
        )["gtvt"]

        selected_slices = g.load_json(os.path.join(patient_dir, "selected_slices.json"))

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
        no_pt: bool,
        no_mr: bool,
        metric_funcs: Dict = None,
        device_id: int = 0,
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
            no_pt=no_pt,
            no_mr=no_mr,
            metric_funcs=metric_funcs,
            idl_gtvt_label_masked_by_selected_slices=idl_gtvt_label_masked_by_selected_slices,
            device_id=device_id,
        )

        # save score of cur patient
        if self._obs_study_progress is None:
            idl_gtvt_dir = Path(round_dir).parent.parent.parent
            score_json_path = os.path.join(
                idl_gtvt_dir, "inference_{}_test.json".format(dataset_ver)
            )
            if os.path.exists(score_json_path):
                score = g.load_json(score_json_path)
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    score[patient][metric][cur_round] = patient_outputs["gtvt"][metric]
                g.save_json(score, score_json_path)

        # save pred of cur patient
        g.save_nii(
            img=patient_outputs["gtvt"]["pred"],
            save_path=os.path.join(round_dir, "gtvt_pred.nii.gz"),
            spacing=g.NII_SPACING,
        )

    def __training_single_round(
        self,
        round_dir: str,
        hyper: Dict,
        selected_slices: Dict,
        metric_funcs: Dict = None,
        queue=None,
        device_id: int = 0,
    ):
        g.create_dir(round_dir)

        cur_round = Path(round_dir).name
        cur_round = int(cur_round[len("round=") :])

        patient_dir = Path(round_dir).parent
        patient = patient_dir.name[len("patient=") :]

        idl_gtvt_dir = patient_dir.parent.parent
        loss_json_path = os.path.join(patient_dir, "loss.json")
        loss_dict = g.load_json(loss_json_path)

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

        # delineation path
        # "delineation_path" is None or not depends on
        # whether "gtvt_delineation.nii.gz" exists or not,
        # and nolonger depends on SelectScenario as SelectScenario
        # is not USER_CLICK when calling obs_study() directly when debugging.
        delineation_path = os.path.join(
            patient_dir, "round=01", "gtvt_delineation.nii.gz"
        )
        if not os.path.exists(delineation_path):
            delineation_path = None

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

        idl_gtvt_dataset = IDLGTVtDataSet(
            patient=patient,
            selected_slices=selected_slices,
            pred_dir=pred_dir,
            delineation_path=delineation_path,  # delineation_path=None for simulation
            dataset_ver=hyper["dataset.ver"],
            no_pt=hyper["no.pt"],
            no_mr=hyper["no.mr"],
            augment=augment,
            weight=weight,
            nr_iters=hyper["iter"],
            use_newcode=True,
        )

        # optimize batch size (before create dataloader)
        self._optimize_batch_size(dataset=idl_gtvt_dataset, hyper=hyper)

        # idl gtvt dataloader
        idl_gtvt_loader = DataLoader(
            dataset=idl_gtvt_dataset,
            batch_size=hyper["batch.size.actual"],
            shuffle=True,
            num_workers=g.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=True,
        )

        if queue is not None:
            progress = 0.02
            queue.put({"progress": progress})
        # if self._obs_study_progress is not None:
        #     self._obs_study_progress.cur_step += (
        #         self._obs_study_progress.step.INIT_DATALOADER
        #     )
        #     self._obs_study_progress.emit_signal()

        # print("iteration_setup",  block_times["iteration_setup"])
        scaler = GradScaler()
        # iter loop through iterations

        cnt = 0
        cur_iter = 0
        batch_count = 0
        augment_times = augment["augment.times"]
        batch_size = hyper["batch.size.actual"]
        idl_gtvt_dataset.set_iter(cur_iter)
        if queue is not None:
            totalsteps = len(idl_gtvt_loader)
            stepsize = (0.9 - progress) / totalsteps
        for item in tqdm(idl_gtvt_loader):  # Iterate through the DataLoader
            if queue is not None:
                progress += stepsize
                queue.put({"progress": progress})
            if cnt == 0:  # Begin a new iteration every batch_size (4 augmentations)
                hyper["cnn"].train()
                iter_loss = 0  # Reset the iteration loss
                idl_gtvt_dataset.set_iter(cur_iter)

                if hyper["layer.freezing"]:
                    if g.used_gpu_count() > 1:
                        hyper["cnn"].module.freeze_top()
                    else:
                        hyper["cnn"].freeze_top()

            hyper["optim"].zero_grad()
            item["input.imgs"] = item["input.imgs"].to(g.DEVICES[device_id])
            item["labels"] = item["labels"].to(g.DEVICES[device_id])
            item["weight.map"] = item["weight.map"].to(g.DEVICES[device_id])

            preds = hyper["cnn"](item["input.imgs"])

            with autocast():
                loss = hyper["loss.func"](preds, item["labels"], item["weight.map"])

            scaler.scale(loss).backward()
            scaler.step(hyper["optim"])
            scaler.update()
            iter_loss += loss.item()
            cnt += batch_size
            batch_count += 1

            if cnt == augment_times:
                # Log and reset after 4 augmentations (1 "iteration")
                iter_loss /= batch_count
                hyper["scheduler"].step(iter_loss)

                # # Emit signal for progress tracking
                # if batch_count == 1 and self._obs_study_progress is not None:
                #     self._obs_study_progress.cur_step += (
                #         self._obs_study_progress.step.FIRST_BATCH
                #     )
                #     self._obs_study_progress.emit_signal()
                # elif self._obs_study_progress is not None:
                #     self._obs_study_progress.cur_step += (
                #         self._obs_study_progress.step.OTHER_BATCH
                #     )
                #     self._obs_study_progress.emit_signal()

                # Store the iteration loss
                loss_dict[
                    "iter={:03d}".format(
                        (cur_round - 1) * hyper["iter"] + (cur_iter + 1)
                    )
                ] = iter_loss
                # Move to the next iteration
                cur_iter += 1
                cnt = 0  # Reset counter for the next iteration
                batch_count = 0
                # Optionally, save after every iteration if there's only one patient
                patient_dirs_list = g.get_sub_dirs(
                    os.path.join(idl_gtvt_dir, "patients")
                )
                if len(patient_dirs_list) <= 1:
                    g.save_json(loss_dict, loss_json_path)
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
            g.save_json(
                data=selected_slices_to_save,
                path=os.path.join(patient_dir, "selected_slices.json"),
            )
        if queue is not None:
            progress += 0.05
            queue.put({"progress": progress})

        # inference
        self.__inference_cur_round(
            round_dir=round_dir,
            cnn=hyper["cnn"],
            dataset_ver=hyper["dataset.ver"],
            no_pt=hyper["no.pt"],
            no_mr=hyper["no.mr"],
            metric_funcs=metric_funcs,
            device_id=device_id,
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
        g.save_json(loss_dict, loss_json_path)
        if queue is not None:
            progress += 0.05
            queue.put({"progress": progress})

    def __simulation_single_patient(
        self,
        patient: str,
        idl_gtvt_dir: str,
        hyper: Dict,
        metric_funcs: Dict = None,
        device_id: int = 0,
    ):
        print("")
        print("patient:", patient)

        # create current patient folder
        patient_dir = os.path.join(
            idl_gtvt_dir, "patients", "patient={}".format(patient)
        )
        g.create_dir(patient_dir)
        # create an empty loss.json
        g.save_json(Dict(), os.path.join(patient_dir, "loss.json"))

        # copy baseline scores
        baseline_score = g.load_json(
            os.path.join(
                Path(idl_gtvt_dir).parent,
                "baseline",
                "inference_{}_test.json".format(hyper["dataset.ver"]),
            )
        )
        idl_gtvt_score = g.load_json(
            os.path.join(
                idl_gtvt_dir,
                "inference_{}_test.json".format(hyper["dataset.ver"]),
            )
        )
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            idl_gtvt_score["patient={}".format(patient)][metric]["round=00"] = (
                baseline_score["patient={}".format(patient)]["gtvt"][metric]
            )
        g.save_json(
            idl_gtvt_score,
            os.path.join(
                idl_gtvt_dir,
                "inference_{}_test.json".format(hyper["dataset.ver"]),
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
                metric_funcs=metric_funcs,
                device_id=device_id,
            )

            if cur_round == max_round:
                break

            # load new lr before next round
            if hyper["lr"]["reset"]:
                self.__load_next_round_lr(cur_round + 1, hyper)

        # draw avg loss of all trained patients
        self._plot_loss_fig(idl_gtvt_dir)

    def plot_loss_fig(self, idl_gtvt_id: str):
        for i in g.get_deep_dirs(g.TRAIN_RESULTS_DIR, key_word=idl_gtvt_id):
            # remove "/" if str endswith it
            if i.endswith("/") or i.endswith("\\"):
                i = i[:-1]
            if i.endswith(idl_gtvt_id):
                idl_gtvt_dir = i
                break
        self._plot_loss_fig(idl_gtvt_dir)

    def _plot_loss_fig(self, idl_gtvt_dir: str):
        # avg loss dict
        avg_loss = Dict()
        for patient_dir in g.get_sub_dirs(
            os.path.join(idl_gtvt_dir, "patients"), full_path=True
        ):
            loss_path = os.path.join(patient_dir, "loss.json")
            if os.path.exists(loss_path):
                cur_patient_loss = g.load_json(loss_path)
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
                avg_loss[i] = g.calculate_avg(avg_loss[i])

            avg_loss = avg_loss.to_list()

            # draw figure
            plt.figure().clear()
            plt.plot(range(1, len(avg_loss) + 1), avg_loss, label="loss")
            plt.legend()
            plt.savefig(os.path.join(idl_gtvt_dir, "loss.png"))

    def simulation(
        self,
        baseline_id: str,
        train_remark: str = None,
        debug_mode: bool = False,
        device_id: int = 0,
    ):
        # load baseline hyper
        self._is_valid_baseline_id(baseline_id)
        baseline_dir = self._find_train_dir(baseline_id)
        if baseline_dir is None:
            g.error_exit("Can not find 'baseline_id'!")
        baseline_fold_dirs = g.get_sub_dirs(
            baseline_dir, key_word="fold=", full_path=True
        )
        baseline_hyper = g.load_json(os.path.join(baseline_fold_dirs[0], "hyper.json"))
        baseline_no_pt = True if baseline_hyper["no.pt"] else False
        baseline_no_mr = True if baseline_hyper["no.mr"] else False

        # load segmentation metrics
        metric_funcs = self._load_metric_funcs(device_id=device_id)

        # load hyper
        hyper_series = self._load_hyper_series_from_json(g.HYPER_PATH["idl.gtvt"])

        for hyper in hyper_series:
            # idl.gtvt doesnt have "no.pt" or "no.mr" hyperparam, copy them from baseline
            hyper["no.pt"] = baseline_no_pt
            hyper["no.mr"] = baseline_no_mr

            idl_gtvt_id = "idl.gtvt_" + self._init_train_id(
                hyper=hyper,
                hyper_json_path=g.HYPER_PATH["idl.gtvt"],
                train_remark=train_remark,
                debug_mode=debug_mode,
            )

            print("")
            print(idl_gtvt_id)

            # create idl result dir
            idl_gtvt_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvt_id)
            g.create_dir(idl_gtvt_dir)

            # load and print hyper
            self._load_hyper(
                hyper=hyper,
                baseline_id=baseline_id,
                debug_mode=debug_mode,
                device_id=device_id,
            )
            print("")
            self._print_hyper(hyper)

            # save hyper before training
            hyper_save_path = os.path.join(idl_gtvt_dir, "hyper.json")
            self._save_hyper(hyper, hyper_save_path)

            # training start time
            hyper["time.spent.total"] = datetime.now()

            # create an empty score json files
            g.save_json(
                Dict(),
                os.path.join(
                    idl_gtvt_dir,
                    "inference_{}_test.json".format(hyper["dataset.ver"]),
                ),
            )

            # best baseline cnn is decided by dataset_part
            baseline_cnn_path = self._find_best_cnn_in_folds(baseline_id)

            patient_list = hyper["patients"][DatasetPart.TEST]

            # loop through each patient
            for patient in patient_list:
                self.__reset_cnn(
                    hyper=hyper,
                    baseline_cnn_path=baseline_cnn_path,
                    device_id=device_id,
                )
                label = g.load_gtv_labels(
                    dataset_ver=hyper["dataset.ver"],
                    patient=patient,
                )["gtvt"]
                if label.max() == 0:
                    continue

                self.__simulation_single_patient(
                    patient=patient,
                    idl_gtvt_dir=idl_gtvt_dir,
                    hyper=hyper,
                    metric_funcs=metric_funcs,
                    device_id=device_id,
                )

                # calculate and save avg and median scores
                self._inference_calculate_avg_median_save_json(
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
        dataset_ver: str,
        patient: str,
        queue: Queue = None,
        debug_mode: bool = False,
        device_id: int = 0,
    ):

        print("")
        print("observer study: {}".format(idl_gtvt_id))

        # load baseline data
        baseline_id = "baseline_obs.study"
        baseline_dir = self._find_train_dir(baseline_id)

        baseline_fold_dirs = g.get_sub_dirs(
            baseline_dir, key_word="fold=", full_path=True
        )
        baseline_hyper = g.load_json(os.path.join(baseline_fold_dirs[0], "hyper.json"))

        if baseline_hyper["no.pt"] is True:
            no_pt = True
        else:
            no_pt = False

        if baseline_hyper["no.mr"] is True:
            no_mr = True
        else:
            no_mr = False

        # load segmentation metrics
        metric_funcs = self._load_metric_funcs(device_id=device_id)

        # create idl result dir
        idl_gtvt_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvt_id)
        g.create_dir(idl_gtvt_dir)

        # create current patient folder
        patient_dir = os.path.join(
            idl_gtvt_dir, "patients", "patient={}".format(patient)
        )
        g.create_dir(patient_dir)

        # load hyper
        hyper_series = self._load_hyper_series_from_json(g.HYPER_PATH["idl.gtvt"])
        hyper = hyper_series[0]

        # idl.gtvt doesnt have "no.pt" or "no.mr" hyperparam, copy them from baseline
        hyper["no.pt"] = no_pt
        hyper["no.mr"] = no_mr

        # change dataset version to input param
        hyper["dataset.ver"] = dataset_ver

        # select scenario
        # (1) in simulation, selected_slices are generated by __select_new_round_slices()
        # and saved to "selected_slices.json" in __training_single_round()
        # (2) in observer study, "selected_slices.json" is already created
        # in the first step: CLICK_GTVT_CENTER
        selected_slices_path = os.path.join(patient_dir, "selected_slices.json")
        if os.path.exists(selected_slices_path):
            hyper["select.scenario"] = SelectScenario.USER_CLICK
        # this happens only when debugging,
        # when obs_study() is called directly (instead of called by a qthread)
        # in this case, simulate user click using gravity center
        else:
            hyper["select.scenario"] = SelectScenario.GRAVITY_CENTER

        # select step
        hyper["select.step.transverse"] = "1"
        hyper["select.step.coronal"] = "1"
        hyper["select.step.sagittal"] = "1"

        # load and print hyper
        self._load_hyper(
            hyper=hyper,
            baseline_id=baseline_id,
            debug_mode=debug_mode,
            device_id=device_id,
        )

        # save hyper before training
        hyper_save_path = os.path.join(idl_gtvt_dir, "hyper.json")
        self._save_hyper(hyper, hyper_save_path)

        # training start time
        hyper["time.spent.total"] = datetime.now()

        # # obs study progress init (after load hyper)
        # if self._obs_study_progress is not None:
        #     self._obs_study_progress.cur_step = 0
        #     dataset_len = hyper["augment.times"]
        #     dataset_len /= hyper["batch.size.actual"]
        #     dataset_len = math.ceil(dataset_len)
        #     self._obs_study_progress.total_step = (
        #         self._obs_study_progress.step.INIT_CNN
        #         + self._obs_study_progress.step.INIT_DATALOADER
        #         + self._obs_study_progress.step.INFERENCE_LOAD_IMG
        #         + self._obs_study_progress.step.INFERENCE_FORWARD
        #     )
        #     self._obs_study_progress.total_step += hyper["iter"] * (
        #         self._obs_study_progress.step.FIRST_BATCH
        #         + self._obs_study_progress.step.OTHER_BATCH * (dataset_len - 1)
        #     )

        # best baseline cnn is decided by dataset_part
        baseline_cnn_path = self._find_best_cnn_in_folds(baseline_id)

        self.__reset_cnn(
            hyper=hyper, baseline_cnn_path=baseline_cnn_path, device_id=device_id
        )

        # idl progress INIT_CNN
        # if self._obs_study_progress is not None:
        #     self._obs_study_progress.cur_step = 0

        # # idl progress INIT_CNN
        # if self._obs_study_progress is not None:
        #     self._obs_study_progress.cur_step += self._obs_study_progress.step.INIT_CNN
        #     self._obs_study_progress.emit_signal()

        # create an empty loss.json
        g.save_json(Dict(), os.path.join(patient_dir, "loss.json"))

        # load selected slices
        if os.path.exists(selected_slices_path):
            selected_slices = g.load_json(
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
            metric_funcs=metric_funcs,
            queue=queue,
            device_id=device_id,
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
        # Signal completion
        if queue is not None:
            queue.put(
                {
                    "status": "complete",
                    "message": "Observer study finished successfully.",
                }
            )

        # if self._obs_study_progress is not None:
        #     print(self._obs_study_progress.cur_step, self._obs_study_progress.total_step)

    def inference_calculate_avg_median_save_json(self, idl_gtvt_id: str):
        print("")
        print("calculate and save avg and median score: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_dir = self._find_train_dir(idl_gtvt_id)
        if idl_gtvt_dir is None:
            g.error_exit("'idl_gtvt_id' not found!")

        hyper = g.load_json(os.path.join(idl_gtvt_dir, "hyper.json"))
        dataset_ver = hyper["dataset.ver"]
        self._is_valid_dataset_ver(dataset_ver)
        print("dataset version: {}".format(dataset_ver))

        self._inference_calculate_avg_median_save_json(
            idl_gtvt_dir=idl_gtvt_dir,
            dataset_ver=dataset_ver,
        )

    def _inference_calculate_avg_median_save_json(
        self,
        idl_gtvt_dir: str,
        dataset_ver: str,
    ):
        score_json_path = os.path.join(
            idl_gtvt_dir, "inference_{}_test.json".format(dataset_ver)
        )
        scores = g.load_json(score_json_path)
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
                scores[Stat.MEDIAN][metric][cur_round] = g.calculate_median(
                    all_patient_scores[metric][cur_round]
                )
                scores[Stat.AVG][metric][cur_round] = g.calculate_avg(
                    all_patient_scores[metric][cur_round]
                )
        g.save_json(data=scores, path=os.path.join(score_json_path))

    def inference(self, idl_gtvt_id: str, debug_mode: bool = False, device_id: int = 0):
        print("")
        print("inference: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_dir = self._find_train_dir(idl_gtvt_id)
        if idl_gtvt_dir is None:
            g.error_exit("'idl_gtvt_id' not found!")

        # load dataset version
        hyper = g.load_json(os.path.join(idl_gtvt_dir, "hyper.json"))
        dataset_ver = hyper["dataset.ver"]
        self._is_valid_dataset_ver(dataset_ver)
        print("dataset version: {}".format(dataset_ver))

        # load no_pt and no_mr
        no_pt = hyper["no.pt"]
        no_mr = hyper["no.mr"]

        # load segmentation metrics
        metric_funcs = self._load_metric_funcs(device_id=device_id)

        # get all patients
        patients = self._load_patients(
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )
        patients = patients[DatasetPart.TEST]

        # copy baseline score
        baseline_score = g.load_json(
            os.path.join(
                Path(idl_gtvt_dir).parent,
                "baseline",
                "inference_{}_test.json".format(dataset_ver),
            )
        )
        idl_gtvt_score_path = os.path.join(
            idl_gtvt_dir, "inference_{}_test.json".format(dataset_ver)
        )
        if os.path.exists(idl_gtvt_score_path):
            idl_gtvt_score = g.load_json(idl_gtvt_score_path)
        else:
            idl_gtvt_score = Dict()
        for patient in patients:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                idl_gtvt_score["patient={}".format(patient)][metric]["round=00"] = (
                    baseline_score["patient={}".format(patient)]["gtvt"][metric]
                )
        g.save_json(idl_gtvt_score, idl_gtvt_score_path)

        # loop through each patient
        for patient in tqdm(patients):
            patient_dir = os.path.join(
                idl_gtvt_dir, "patients", "patient={}".format(patient)
            )

            # loop through each round
            for round_dir in g.get_sub_dirs(
                patient_dir, key_word="round=", full_path=True
            ):
                # load current round cnn
                cnn_path = g.get_sub_files(round_dir, key_word=".pt", full_path=True)[0]
                cnn = self._load_exist_cnn(cnn_path, device_id=device_id)

                self.__inference_cur_round(
                    round_dir=round_dir,
                    cnn=cnn,
                    dataset_ver=dataset_ver,
                    no_pt=no_pt,
                    no_mr=no_mr,
                    metric_funcs=metric_funcs,
                    device_id=device_id,
                )

        self._inference_calculate_avg_median_save_json(
            idl_gtvt_dir=idl_gtvt_dir,
            dataset_ver=dataset_ver,
        )

    def _inference_single_patient_record_labels(
        self,
        outputs: Dict,
        dataset_item: Dict,
    ):
        labels = dataset_item["labels"][1].cpu().numpy()
        img_shape = dataset_item["shape"]
        labels = g.center_align_img(labels, img_shape)

        outputs["gtvt"]["label"] = labels
        return outputs

    def _inference_single_patient_record_preds(
        self,
        outputs: Dict,
        preds: ndarray,
        img_shape: tuple,
    ):
        # preds: [background, gtvt]
        outputs["gtvt"]["pred"] = g.center_align_img(preds[1], img_shape)

    # remove connected_components has no overlap with delineated slices
    def _inference_single_patient_gtvt_post_process(
        self,
        outputs: Dict,
        idl_gtvt_label_masked_by_selected_slices: ndarray,
    ):
        if idl_gtvt_label_masked_by_selected_slices is not None:
            cc_list = g.get_connected_components(outputs["gtvt"]["pred"])
            outputs["gtvt"]["pred"] = np.zeros_like(outputs["gtvt"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * idl_gtvt_label_masked_by_selected_slices).sum() > 0:
                    outputs["gtvt"]["pred"] = np.maximum(
                        outputs["gtvt"]["pred"], cur_cc
                    )

    def _is_valid_idl_dataset_ver(
        self,
        hyper: Dict,
        baseline_dataset_ver: str,
    ):
        # baseline trained with or without pt
        if not hyper["no.pt"] and hyper["dataset.ver"] == DatasetVer.MDA:
            g.error_exit("idl.gtvt on mda requires a baseline trained without pet.")

        # baseline trained with or without mr
        if not hyper["no.mr"] and hyper["dataset.ver"] == DatasetVer.HECKTOR:
            g.error_exit("idl.gtvt on hecktor requires a baseline trained without mr.")

        if baseline_dataset_ver not in [
            DatasetVer.AU,
            DatasetVer.MDA,
            DatasetVer.NKI,
            DatasetVer.HECKTOR,
        ]:
            g.error_exit(ErrMsg.DATASET_VER_INVALID)

        # baseline trained on AU dataset
        elif baseline_dataset_ver == DatasetVer.AU:
            self._is_valid_dataset_ver(hyper["dataset.ver"])

        # baseline trained on other datasets
        else:
            if hyper["dataset.ver"] != baseline_dataset_ver:
                g.error_exit(ErrMsg.DATASET_VER_INVALID)
