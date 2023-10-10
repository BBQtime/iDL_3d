import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from custom import (GPU, DatasetPart, DatasetVer, Debug, Dict, DirExplorer,
                    Folder)
from custom import Global as g
from custom import Img, Json, List, Metric, Nii, Plane, Value
from dataset_idl_gtvt import DataSetIDLGTVt
from loss_func_idl_gtvt import UnifiedFocalLossIDLGTVt
from numpy import ndarray
from scipy.ndimage import measurements
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_core import TrainingCore


class TrainingIDLGTVt(TrainingCore):
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
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
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

    # def real_training(
    #     self,
    #     baseline_id: str,
    #     idl_results_dir: str,
    #     idl_gtvt_id: str,
    #     cur_patient: str,
    #     round_num: int,
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
    #     cur_result_dir = os.path.join(idl_results_dir, self._idl_id)
    #     if not os.path.exists(cur_result_dir):
    #         print("TrainingIDLGTVt.real_training(): iDL result folder doesn't exist")

    #     # create json file to save train loss
    #     train_loss_dict = Dict()
    #     train_loss_dict["iter"] = Dict()
    #     Json.save(
    #         train_loss_dict,
    #         os.path.join(
    #             cur_result_dir, "patient={}".format(cur_patient), "train_loss.json"
    #         ),
    #     )

    #     # get annotated slices
    #     round_dir = os.path.join(
    #         cur_result_dir,
    #         "patient={}".format(cur_patient),
    #         "round={:02d}".format(round_num),
    #     )
    #     selected_slices = Dict()
    #     selected_slices["round=01"] = List()  # doesn't matter what the dict key is
    #     for file_name in DirExplorer.get_sub_files(round_dir, key_word="_label.npy"):
    #         slice_id = file_name[len("slice_") : -len("_label.npy")]
    #         slice_id = slice_id.zfill(3)
    #         selected_slices["round=01"].append(slice_id)

    #     # training start time
    #     hyper["time.spent"] = datetime.now()

    #     self.__training_single_round(
    #         cur_result_dir=cur_result_dir,
    #         cur_patient=cur_patient,
    #         selected_slices=selected_slices,
    #         label_dir=round_dir,
    #     )

    #     # get training time spent before save hyper
    #     hyper["time.spent"] = datetime.now() - hyper["time.spent"]
    #     # save hyper
    #     self._save_hyper(os.path.join(cur_result_dir, "hyper.json"))

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

        round_num = max(
            len(selected_slices[Plane.TRANSVERSE]),
            len(selected_slices[Plane.CORONAL]),
            len(selected_slices[Plane.SAGITTAL]),
        )
        round_num += 1

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
            if len(hyper["select.step.{}".format(plane)]) < round_num:
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
                # skip slice that already been annotated
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
            if hyper["select.scenario"] == "largest":
                # descrease sort the dict (return a list of tuple)
                candidates = candidates.sort_by_value(reverse=True)
                new_round_slices[plane] = candidates.keys()

            # "gravity.center", round = 1
            elif hyper["select.scenario"] == "gravity.center" and round_num == 1:
                if plane == Plane.TRANSVERSE:
                    new_round_slices[plane].append(round(label_center[0]))
                elif plane == Plane.CORONAL:
                    new_round_slices[plane].append(round(label_center[1]))
                elif plane == Plane.SAGITTAL:
                    new_round_slices[plane].append(round(label_center[2]))

            # "equal.divide", round = 1
            elif hyper["select.scenario"] == "equal.divide" and round_num == 1:
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
            if hyper["select.scenario"] == "gravity.center" and round_num == 1:
                new_slices_num = 1
            else:
                new_slices_num = hyper["select.step.{}".format(plane)][round_num - 1]
            if new_slices_num < len(new_round_slices[plane]):
                new_round_slices[plane] = new_round_slices[plane][:new_slices_num]

            # add new_round_slices into selected_slices
            selected_slices[plane]["round={:02d}".format(round_num)] = new_round_slices[
                plane
            ]

        return new_round_slices

    def __get_masked_label(self, round_dir: str, dataset_ver: str):
        round_num = Path(round_dir).name
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
        slice_mask = Dict()

        # loop through each plane
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            slice_mask[plane] = np.zeros(label.shape, dtype=np.float32)

            # loop through each round
            for round_num in selected_slices[plane]:
                # str to list
                selected_slices[plane][round_num] = List(
                    selected_slices[plane][round_num]
                )
                # current step
                for slice_num in selected_slices[plane][round_num]:
                    # change slice id from str into int
                    slice_num = int(slice_num)
                    if plane == Plane.TRANSVERSE:
                        slice_mask[plane][slice_num, :, :] = np.ones_like(
                            slice_mask[plane][0, :, :]
                        )
                    elif plane == Plane.CORONAL:
                        slice_mask[plane][:, slice_num, :] = np.ones_like(
                            slice_mask[plane][:, 0, :]
                        )
                    elif plane == Plane.SAGITTAL:
                        slice_mask[plane][:, :, slice_num] = np.ones_like(
                            slice_mask[plane][:, :, 0]
                        )

        # combine slice_mask on 3 planes
        slice_mask = np.maximum(
            np.maximum(slice_mask[Plane.TRANSVERSE], slice_mask[Plane.CORONAL]),
            slice_mask[Plane.SAGITTAL],
        )
        label *= slice_mask
        return label

    def __inference_round(
        self,
        round_dir: str,
        cnn,
        dataset_ver: str,
        dataset_part: str,
        no_pt: str,
        segment_metrics,
    ):
        round_num = Path(round_dir).name

        patient = Path(round_dir).parent.name

        # get annotation for post processing
        idl_gtvt_masked_label = self.__get_masked_label(
            round_dir=round_dir, dataset_ver=dataset_ver
        )

        # result structure: gtvt: {pred, dsc, msd, hd95}
        patient_outputs = self._inference_single_patient(
            patient=patient[len("patient=") :],
            cnn=cnn,
            dataset_ver=dataset_ver,
            dataset_part=dataset_part,
            no_pt=no_pt,
            segment_metrics=segment_metrics,
            idl_gtvt_masked_label=idl_gtvt_masked_label,
        )

        # save score of cur patient
        idl_gtvt_dir = Path(round_dir).parent.parent.parent
        score_json_path = os.path.join(
            idl_gtvt_dir, "inference_{}_{}.json".format(dataset_ver, dataset_part)
        )
        score = Json.load(score_json_path)
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            score[patient][metric][round_num] = patient_outputs["gtvt"][metric]
        Json.save(score, score_json_path)

        # save pred of cur patient
        Nii.save(
            img=patient_outputs["gtvt"]["pred"],
            save_path=os.path.join(round_dir, "gtvt_pred.nii"),
            spacing=g.NII_SPACING[dataset_ver],
        )

    def __training_single_round(
        self,
        round_dir: str,
        label_dir: str,
        hyper: Dict,
        selected_slices: Dict,
        segment_metrics: Dict,
        dataset_part: str,
    ):
        Folder.create(round_dir)

        round_num = Path(round_dir).name
        round_num = int(round_num[len("round=") :])

        patient_dir = Path(round_dir).parent
        patient = patient_dir.name[len("patient=") :]

        idl_gtvt_dir = patient_dir.parent.parent
        loss_json_path = os.path.join(patient_dir, "loss.json")
        loss_dict = Json.load(loss_json_path)

        if round_num == 1:
            pred_dir = os.path.join(
                idl_gtvt_dir.parent,
                "baseline",
                "patients",
                "patient={}".format(patient),
            )
        else:
            pred_dir = os.path.join(patient_dir, "round={:02d}".format(round_num - 1))

        # record current round time spent
        time_spent = datetime.now()

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

        dataset_idl_gtvt = DataSetIDLGTVt(
            patient=patient,
            selected_slices=selected_slices,
            label_dir=label_dir,
            pred_dir=pred_dir,
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

        # iter loop
        for iter_num in tqdm(range(hyper["iter"])):
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

            # cur iter finished
            # update scheduler
            iter_loss /= batch_count
            hyper["scheduler"].step(iter_loss)

            # record loss
            loss_dict[
                "iter={:03d}".format((round_num - 1) * hyper["iter"] + (iter_num + 1))
            ] = iter_loss
            # save loss and update loss figure after every iter, if there is only one patient
            patient_dirs_list = DirExplorer.get_sub_folders(
                os.path.join(idl_gtvt_dir, "patients")
            )
            if len(patient_dirs_list) <= 1:
                Json.save(loss_dict, loss_json_path)
                self._plot_loss_fig(idl_gtvt_dir)

        # current round idl finished
        # save cnn
        cnn_save_path = os.path.join(round_dir, Path(round_dir).name + ".pt")
        self._save_cnn(hyper, cnn_save_path)

        # save selected_slices dict before inference, because masked_label needs it
        # copy a new dict to avoid changing origin selected_slices dict
        selected_slices_to_save = selected_slices.copy()
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            for round_num in selected_slices_to_save[plane]:
                selected_slices_to_save[plane][round_num] = selected_slices_to_save[
                    plane
                ][round_num].to_str()
        Json.save(
            data=selected_slices_to_save,
            path=os.path.join(patient_dir, "selected_slices.json"),
        )

        # inference
        self.__inference_round(
            round_dir=round_dir,
            cnn=hyper["cnn"],
            dataset_ver=hyper["dataset.ver"],
            dataset_part=dataset_part,
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

    def __training_single_patient(
        self,
        patient: str,
        idl_gtvt_dir: str,
        hyper: Dict,
        segment_metrics: Dict,
        dataset_part: str,
    ):
        print("")
        print("patient:", patient)

        # create current patient folder
        patient_dir = os.path.join(
            idl_gtvt_dir, "patients", "patient={}".format(patient)
        )
        Folder.create(patient_dir)
        # create an empty loss.json
        Json.save(Dict(), os.path.join(patient_dir, "loss.json"))

        # copy baseline scores
        baseline_score = Json.load(
            os.path.join(
                Path(idl_gtvt_dir).parent,
                "baseline",
                "inference_{}_{}.json".format(hyper["dataset.ver"], dataset_part),
            )
        )
        idl_gtvt_score = Json.load(
            os.path.join(
                idl_gtvt_dir,
                "inference_{}_{}.json".format(hyper["dataset.ver"], dataset_part),
            )
        )
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            idl_gtvt_score["patient={}".format(patient)][metric][
                "round=00"
            ] = baseline_score["patient={}".format(patient)]["gtvt"][metric]
        Json.save(
            idl_gtvt_score,
            os.path.join(
                idl_gtvt_dir,
                "inference_{}_{}.json".format(hyper["dataset.ver"], dataset_part),
            ),
        )

        selected_slices = Dict()

        # loop through each round
        max_round = max(
            len(hyper["select.step.transverse"]),
            len(hyper["select.step.coronal"]),
            len(hyper["select.step.sagittal"]),
        )
        for round_num in range(1, max_round + 1):
            # new_round_slices are add into selected_slices in this function
            new_round_slices = self.__select_new_round_slices(
                selected_slices=selected_slices,
                hyper=hyper,
                patient_dir=patient_dir,
            )

            # no slice needs to be annotated in cur round
            if (
                len(new_round_slices[Plane.TRANSVERSE]) == 0
                and len(new_round_slices[Plane.CORONAL]) == 0
                and len(new_round_slices[Plane.SAGITTAL]) == 0
            ):
                break

            # start current round
            print("round:", round_num)

            round_dir = os.path.join(patient_dir, "round={:02d}".format(round_num))
            self.__training_single_round(
                round_dir=round_dir,
                label_dir=g.DATASET_DIR[hyper["dataset.ver"]],
                hyper=hyper,
                selected_slices=selected_slices,
                segment_metrics=segment_metrics,
                dataset_part=dataset_part,
            )

            if round_num == max_round:
                break

            # load new lr before next round
            if hyper["lr"]["reset"]:
                self.__load_next_round_lr(round_num + 1, hyper)

        # draw avg loss of all trained patients
        self._plot_loss_fig(idl_gtvt_dir)

    def plot_loss_fig(self, idl_gtvt_id: str):
        for i in DirExplorer.walk_sub_dirs(g.TRAIN_RESULTS_DIR, key_word=idl_gtvt_id):
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
        for patient_dir in DirExplorer.get_sub_folders(
            os.path.join(idl_gtvt_dir, "patients"), full_path=True
        ):
            cur_patient_loss = Json.load(os.path.join(patient_dir, "loss.json"))
            if avg_loss == {}:
                for i in cur_patient_loss:
                    avg_loss[i] = [cur_patient_loss[i]]
            else:
                for i in avg_loss:
                    avg_loss[i].append(cur_patient_loss[i])

        for i in avg_loss:
            avg_loss[i] = Value.avg(avg_loss[i])

        avg_loss = avg_loss.to_list()

        # draw figure
        plt.figure().clear()
        plt.plot(range(1, len(avg_loss) + 1), avg_loss, label="loss")
        plt.legend()
        plt.savefig(os.path.join(idl_gtvt_dir, "loss.png"))

    def __find_best_baseline_fold_cnn(
        self, hyper: Dict, baseline_id: str, dataset_part: str
    ) -> str:
        scores = Dict()

        fold_dirs = DirExplorer.get_sub_folders(
            input_dir=os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, "baseline"),
            key_word="fold=",
            full_path=True,
        )
        for fold_dir in fold_dirs:
            fold = Path(fold_dir).name
            epoch_dir = DirExplorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            )[0]
            epoch_scores = Json.load(
                os.path.join(
                    epoch_dir,
                    "inference_{}_{}.json".format(hyper["dataset.ver"], dataset_part),
                )
            )
            for stats in ["median", "avg"]:
                scores[fold][stats] = epoch_scores[stats]

        for stats in ["median", "avg"]:
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    # create a tmp list to sort
                    list_to_sort = List()
                    # add elements into the list
                    for epoch in scores.keys():
                        list_to_sort.append(scores[epoch][stats][gtv][metric])
                    # sort the list
                    if metric == Metric.DSC:
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)
                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        new_value = list_to_sort.index(
                            scores[epoch][stats][gtv][metric]
                        )
                        # if metric == Metric.DSC:
                        #     new_value *= 2
                        scores[epoch][stats][gtv][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stats in ["avg", "median"]:
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                        evaluation[epoch] += scores[epoch][stats][gtv][metric]

        best_fold = evaluation.key_with_max_value()
        best_fold_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, baseline_id, "baseline", best_fold
        )
        best_epoch_dir = DirExplorer.get_sub_folders(
            best_fold_dir, key_word="epoch=", full_path=True
        )[0]
        best_cnn_path = DirExplorer.get_sub_files(
            best_epoch_dir, key_word=".pt", full_path=True
        )[0]
        return best_cnn_path

    def simulation(
        self,
        baseline_id: str,
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        self._is_valid_baseline_id(baseline_id)

        baseline_dir = self._find_train_dir(baseline_id)
        if baseline_dir is None:
            Debug.error_exit("training id not found")

        baseline_fold_dirs = DirExplorer.get_sub_folders(
            baseline_dir, key_word="fold=", full_path=True
        )

        baseline_hyper = Json.load(os.path.join(baseline_fold_dirs[0], "hyper.json"))
        no_pt = baseline_hyper["no.pt"]
        baseline_dataset_ver = baseline_hyper["dataset.ver"]

        dataset_ver = self._is_valid_dataset_version(
            dataset_ver=dataset_ver,
            origin_dataset_ver=baseline_dataset_ver,
        )
        if dataset_ver == DatasetVer.AU_3MM or dataset_ver == DatasetVer.AU_1MM:
            dataset_part_list = [DatasetPart.TEST_INTER]  # , DatasetPart.TEST_EXTER]
        elif dataset_ver == DatasetVer.MDA:
            dataset_part_list = [DatasetPart.TEST]

        # print("dataset version: {}".format(dataset_ver))
        # print("dataset section: {}".format(dataset_part))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics(dataset_ver)

        for hyper in self._load_hyper_series_from_json(g.HYPER_JSON_PATH["idl.gtvt"]):
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

            # load and print hyper
            self._load_hyper(
                hyper=hyper,
                baseline_id=baseline_id,
                debug_mode=debug_mode,
            )
            print("")
            self._print_hyper(hyper)

            # create idl result dir
            idl_gtvt_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvt_id)
            Folder.create(idl_gtvt_dir)

            # save hyper before training
            hyper_save_path = os.path.join(idl_gtvt_dir, "hyper.json")
            self._save_hyper(hyper, hyper_save_path)

            # training start time
            hyper["time.spent.total"] = datetime.now()

            for dataset_part in dataset_part_list:
                self._is_valid_dataset_part(
                    dataset_part=dataset_part,
                    dataset_ver=dataset_ver,
                )
                # create an empty score json files
                Json.save(
                    Dict(),
                    os.path.join(
                        idl_gtvt_dir,
                        "inference_{}_{}.json".format(
                            hyper["dataset.ver"], dataset_part
                        ),
                    ),
                )

                # best baseline cnn is decided by dataset_part
                baseline_cnn_path = self.__find_best_baseline_fold_cnn(
                    hyper=hyper,
                    baseline_id=baseline_id,
                    dataset_part=dataset_part,
                )

                # loop through each patient
                patient_list = hyper["patients"][dataset_part]
                for patient in patient_list:
                    self.__reset_cnn(
                        hyper=hyper,
                        baseline_cnn_path=baseline_cnn_path,
                    )
                    self.__training_single_patient(
                        patient=patient,
                        idl_gtvt_dir=idl_gtvt_dir,
                        hyper=hyper,
                        segment_metrics=segment_metrics,
                        dataset_part=dataset_part,
                    )
                    self._inference_calculate_save_avg_median(
                        idl_gtvt_dir,
                        dataset_ver=hyper["dataset.ver"],
                        dataset_part=dataset_part,
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

    def inference_calculate_save_avg_median(self, idl_gtvt_id: str, dataset_part: str):
        print("")
        print("calculate and save avg and median score: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_dir = self._find_train_dir(idl_gtvt_id)
        if idl_gtvt_dir is None:
            Debug.error_exit("idl_gtvt_id not found")

        hyper = Json.load(os.path.join(idl_gtvt_dir, "hyper.json"))
        dataset_ver = hyper["dataset.ver"]
        dataset_ver = self._is_valid_dataset_version(dataset_ver=dataset_ver)
        self._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))
        print("dataset section: {}".format(dataset_part))

        self._inference_calculate_save_avg_median(
            idl_gtvt_dir=idl_gtvt_dir,
            dataset_ver=dataset_ver,
            dataset_part=dataset_part,
        )

    def _inference_calculate_save_avg_median(
        self, idl_gtvt_dir: str, dataset_ver: str, dataset_part: str
    ):
        score_json_path = os.path.join(
            idl_gtvt_dir, "inference_{}_{}.json".format(dataset_ver, dataset_part)
        )
        scores = Json.load(score_json_path)
        all_patient_scores = Dict()

        # add all patients score in to a list
        for patient in scores:
            if patient == "median" or patient == "avg":
                continue
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                for round_num in scores[patient][metric]:
                    if all_patient_scores[metric][round_num] == {}:
                        all_patient_scores[metric][round_num] = List()
                    all_patient_scores[metric][round_num].append(
                        scores[patient][metric][round_num]
                    )
        # calculate median score
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            for round_num in all_patient_scores[metric]:
                scores["median"][metric][round_num] = Value.median(
                    all_patient_scores[metric][round_num]
                )
                scores["avg"][metric][round_num] = Value.avg(
                    all_patient_scores[metric][round_num]
                )
        Json.save(data=scores, path=os.path.join(score_json_path))

    def inference(self, idl_gtvt_id: str, dataset_part: str, debug_mode: bool = False):
        print("")
        print("inference: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        idl_gtvt_dir = self._find_train_dir(idl_gtvt_id)
        if idl_gtvt_dir is None:
            Debug.error_exit("idl_gtvt_id not found")

        # load dataset version
        hyper = Json.load(os.path.join(idl_gtvt_dir, "hyper.json"))
        dataset_ver = hyper["dataset.ver"]
        no_pt = hyper["no.pt"]
        dataset_ver = self._is_valid_dataset_version(dataset_ver=dataset_ver)
        self._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))
        print("dataset section: {}".format(dataset_part))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics(dataset_ver)

        # get all patients
        patients = self._load_patients(
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )
        patients = patients[dataset_part]

        # copy baseline score
        baseline_score = Json.load(
            os.path.join(
                Path(idl_gtvt_dir).parent,
                "baseline",
                "inference_{}_{}.json".format(dataset_ver, dataset_part),
            )
        )
        idl_gtvt_score_path = os.path.join(
            idl_gtvt_dir, "inference_{}_{}.json".format(dataset_ver, dataset_part)
        )
        if os.path.exists(idl_gtvt_score_path):
            idl_gtvt_score = Json.load(idl_gtvt_score_path)
        else:
            idl_gtvt_score = Dict()
        for patient in patients:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                idl_gtvt_score["patient={}".format(patient)][metric][
                    "round=00"
                ] = baseline_score["patient={}".format(patient)]["gtvt"][metric]
        Json.save(idl_gtvt_score, idl_gtvt_score_path)

        # loop through each patient
        for patient in tqdm(patients):
            patient_dir = os.path.join(
                idl_gtvt_dir, "patients", "patient={}".format(patient)
            )

            # loop through each round
            for round_dir in DirExplorer.get_sub_folders(
                patient_dir, key_word="round=", full_path=True
            ):
                # load current round cnn
                cnn_path = DirExplorer.get_sub_files(
                    round_dir, key_word=".pt", full_path=True
                )[0]
                cnn = self._load_exist_cnn(cnn_path)

                self.__inference_round(
                    round_dir=round_dir,
                    cnn=cnn,
                    dataset_ver=dataset_ver,
                    dataset_part=dataset_part,
                    no_pt=no_pt,
                    segment_metrics=segment_metrics,
                )

        self._inference_calculate_save_avg_median(
            idl_gtvt_dir=idl_gtvt_dir,
            dataset_ver=dataset_ver,
            dataset_part=dataset_part,
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
        self, outputs: Dict, idl_gtvt_masked_label: ndarray
    ):
        if idl_gtvt_masked_label is not None:
            cc_list = Img.connected_components(outputs["gtvt"]["pred"])
            outputs["gtvt"]["pred"] = np.zeros_like(outputs["gtvt"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * idl_gtvt_masked_label).sum() > 0:
                    outputs["gtvt"]["pred"] = np.maximum(
                        outputs["gtvt"]["pred"], cur_cc
                    )

    def _is_valid_dataset_part(
        self,
        dataset_part: str,
        dataset_ver: str = None,
    ):
        if dataset_part not in [
            DatasetPart.TEST,
            DatasetPart.TEST_INTER,
            DatasetPart.TEST_EXTER,
        ]:
            Debug.error_exit(
                "'dataset_part' can not take on any values other than 'test/test.inter/test.exter'"
            )

        super()._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )
