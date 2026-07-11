import os
from pathlib import Path
from typing import Any

import global_utils.global_core as g
import numpy as np
import torch
from dataset_utils.idl_gtvn_dataset import IDLGTVnDataSet
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetPart, DatasetVer, ErrMsg, Metric, Stats
from loss_utils.idl_gtvn_loss import IDLGTVnLoss
from numpy import ndarray
from torch import Tensor
from training_utils.baseline_training import BaselineTraining
from training_utils.training_core import ObsStudyProgress


class GTVnObsStudyProgress(ObsStudyProgress):
    class ProgressStep:
        INFERENCE_INIT = 1
        INFERENCE_LOAD_IMG = 3
        INFERENCE_FORWARD = 1
        INFERENCE_SAVE_PRED = 1
        CROSS_VALID = 3

    def __init__(self):
        super().__init__()
        self.step = self.ProgressStep()


class IDLGTVnTraining(BaselineTraining):
    def __init__(self, idl_progress_signal: Any = None):
        super().__init__()
        if idl_progress_signal is not None:
            self._obs_study_progress = GTVnObsStudyProgress()
            self._obs_study_progress.progress_signal = idl_progress_signal
        else:
            self._obs_study_progress = None

    def _load_hyper_new_cnn(
        self,
        hyper: Dict,
        device_id: int,
        in_chan: int = 5,
        out_chan: int = 2,
    ):
        # No need to reduce CNN input channels here if no_pt is True.
        # The input channels will be reduced in super()._load_hyper_new_cnn instead.
        super()._load_hyper_new_cnn(
            hyper=hyper,
            device_id=device_id,
            in_chan=in_chan,
            out_chan=out_chan,
        )

    def _load_hyper_loss_func(self, hyper: Dict, device_id: int):
        hyper["loss.func"] = IDLGTVnLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.get_device(device_id))

    def _load_hyper_data_sets(self, hyper: Dict):
        # load train/valid/test datasets
        for i in [DatasetPart.TRAIN, DatasetPart.VALID]:
            # only use data augmentation on training set
            if i == DatasetPart.TRAIN:
                augment = Dict()
                augment["augment.methods"] = hyper["augment.methods"]
                augment["augment.pct"] = hyper["augment.pct"]
                augment["augment.min"] = hyper["augment.min"]
                augment["augment.max"] = hyper["augment.max"]
            else:
                augment = None
            hyper["{}.set".format(i)] = IDLGTVnDataSet(
                patients=hyper["{}.patients".format(i)],
                dataset_ver=hyper["dataset.ver"],
                no_pt=hyper["no.pt"],
                no_mr=hyper["no.mr"],
                geodesic_distance=hyper["geodesic.distance"],
                augment=augment,
                random_click=False,
            )

    def new_training(
        self,
        baseline_id: str,
        train_remark: str = "",
        device_id: int = -1,  # use all cards by default
        debug_mode: bool = False,
    ):
        self._is_valid_baseline_id(baseline_id)
        self._new_training(
            idl_gtvn_baseline_id=baseline_id,
            train_remark=train_remark,
            device_id=device_id,
            debug_mode=debug_mode,
        )

    def obs_study(
        self,
        idl_gtvn_id: str,
        dataset_ver: str,
        patient: str,
        obs_gtvn_clicks: ndarray = None,  # None means no gtvn click
        device_id: int = 0,  # Use card 0 by default, as GTVn inference requires less resources than GTVt re-training.
        debug_mode=False,
    ):
        print("")
        print("observer study: {}".format(idl_gtvn_id))

        baseline_id = "baseline_obs.study"
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)

        obs_study_cnns_base_dir = os.path.join(baseline_dir, "idl.gtvn_obs.study")
        if not os.path.exists(obs_study_cnns_base_dir):
            g.error_exit("'idl.gtvn_obs.study' folder not found!")

        obs_study_output_dir = os.path.join(baseline_dir, idl_gtvn_id)
        if not os.path.exists(obs_study_output_dir):
            g.create_dir(obs_study_output_dir)

        cnn_fold_dirs = g.get_sub_dirs(
            obs_study_cnns_base_dir, key_word="fold=", full_path=True
        )
        hyper = g.load_json(os.path.join(cnn_fold_dirs[0], "hyper.json"))
        no_pt = hyper["no.pt"]
        no_mr = hyper["no.mr"]
        geodesic_distance = bool(hyper["geodesic.distance"])

        # load segmentation metrics
        metric_funcs = self._load_metric_funcs(device_id)

        # idl progress init
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step = 0
            if debug_mode:
                self._obs_study_progress.total_step = (
                    self._obs_study_progress.step.INFERENCE_INIT
                    + self._obs_study_progress.step.INFERENCE_LOAD_IMG
                    + self._obs_study_progress.step.INFERENCE_FORWARD
                    + self._obs_study_progress.step.INFERENCE_SAVE_PRED
                    + self._obs_study_progress.step.CROSS_VALID
                )
            else:
                self._obs_study_progress.total_step = (
                    self._obs_study_progress.step.INFERENCE_INIT
                    + self._obs_study_progress.step.INFERENCE_LOAD_IMG
                    + self._obs_study_progress.step.INFERENCE_FORWARD
                    + self._obs_study_progress.step.INFERENCE_SAVE_PRED
                ) * len(cnn_fold_dirs) + self._obs_study_progress.step.CROSS_VALID

        input_imgs, img_shape, dataset_item = (
            self._prepare_for_inference_single_patient(
                patient=patient,
                dataset_ver=dataset_ver,
                no_pt=no_pt,
                no_mr=no_mr,
                device_id=device_id,
                idl_gtvn_geodesic_distance=geodesic_distance,
                obs_gtvn_clicks=obs_gtvn_clicks,
            )
        )
        # loop through fold dirs
        for cnn_fold_dir in cnn_fold_dirs:
            output_fold_dir = os.path.join(
                obs_study_output_dir, Path(cnn_fold_dir).name
            )
            g.create_dir(output_fold_dir)

            cnn_epoch_dir = g.get_sub_dirs(
                cnn_fold_dir, key_word="epoch=", full_path=True
            )[0]
            epoch = int(Path(cnn_epoch_dir).name[len("epoch=") :])

            output_epoch_dir = os.path.join(output_fold_dir, Path(cnn_epoch_dir).name)
            g.create_dir(output_epoch_dir)

            # load cnn
            cnn_path = os.path.join(cnn_epoch_dir, "epoch={:03d}.pt".format(epoch))
            cnn = self._load_exist_cnn(cnn_path, device_id=device_id)

            # idl progress INFERENCE_INIT
            if self._obs_study_progress is not None:
                self._obs_study_progress.cur_step += (
                    self._obs_study_progress.step.INFERENCE_INIT
                )
                self._obs_study_progress.emit_signal()

            patient_outputs = self._inference_single_prepared_patient(
                cnn=cnn,
                dataset_item=dataset_item,
                input_imgs=input_imgs,
                img_shape=img_shape,
                metric_funcs=metric_funcs,
            )
            # # outputs structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
            # patient_outputs = self._inference_single_patient(
            #     patient=patient,
            #     cnn=cnn,
            #     dataset_ver=dataset_ver,
            #     no_pt=no_pt,
            #     no_mr=no_mr,
            #     metric_funcs=metric_funcs,
            #     idl_gtvn_geodesic_distance=geodesic_distance,
            #     obs_gtvn_clicks=obs_gtvn_clicks,  # this is only for obs study
            #     device_id=device_id,
            # )

            # create folder and save preds of current patient
            self._inference_all_folds_save_patient_preds(
                patient=patient,
                epoch_dir=output_epoch_dir,
                patient_outputs=patient_outputs,
            )

            # idl progress INFERENCE_SAVE_PRED
            if self._obs_study_progress is not None:
                self._obs_study_progress.cur_step += (
                    self._obs_study_progress.step.INFERENCE_SAVE_PRED
                )
                self._obs_study_progress.emit_signal()

            # only run 1 fold in debugging mode
            if debug_mode:
                break

        # cross valid
        # initialize preds
        preds = Dict()
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            preds[gtv] = None

        output_fold_dirs = g.get_sub_dirs(
            obs_study_output_dir, key_word="fold=", full_path=True
        )
        for output_fold_dir in output_fold_dirs:
            # find epoch dir
            output_epoch_dir = g.get_sub_dirs(
                output_fold_dir, key_word="epoch=", full_path=True
            )[0]

            # load preds
            output_patient_dir = os.path.join(
                output_epoch_dir, "patients", "patient={}".format(patient)
            )
            for gtv in ["gtvt", "gtvn"]:
                pred_path = os.path.join(
                    output_patient_dir, "{}_pred.nii.gz".format(gtv)
                )
                if os.path.exists(pred_path):
                    img = g.load_nii(path=pred_path, binary=False)
                    if preds[gtv] is None:
                        preds[gtv] = img
                    else:
                        preds[gtv] += img

        # all folds is traversed
        # for idl.gtvn pred["gtvt"] will be None
        if preds["gtvt"] is not None:
            preds["gtvs"] = preds["gtvt"] + preds["gtvn"]

        for gtv in preds.keys():
            if preds[gtv] is None:
                preds.pop(gtv)
            else:
                preds[gtv] /= len(output_fold_dirs)

        # create cross_valid dir
        pred_dir = os.path.join(
            obs_study_output_dir, "patients", "patient={}".format(patient)
        )
        pred_dir = os.path.join(pred_dir, "round=01")
        g.create_dir(pred_dir)

        # save cross_valid preds (only save gtvt and gtvn)
        for gtv in preds.keys():
            if gtv != "gtvs":
                g.save_nii(
                    img=preds[gtv],
                    save_path=os.path.join(pred_dir, "{}_pred.nii.gz".format(gtv)),
                    spacing=g.NII_SPACING,
                )

        # idl progress CROSS_VALID
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step += (
                self._obs_study_progress.step.CROSS_VALID
            )
            self._obs_study_progress.emit_signal()

        g.clear_gpu_cache()
        g.clear_linux_trash()
        if not debug_mode:
            g.clear_debug_data()

        # if self._obs_study_progress is not None:
        #     print(self._obs_study_progress.cur_step, self._obs_study_progress.total_step)

    def inference_all_folds(
        self,
        idl_gtvn_id: str,
        dataset_ver: str = None,
        dataset_part: str = DatasetPart.TEST,  # only valid or test
        device_id: int = 1,  # use card 1 by default
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._inference_all_folds(
            train_id=idl_gtvn_id,
            dataset_ver=dataset_ver,
            dataset_part=dataset_part,
            device_id=device_id,
            debug_mode=debug_mode,
        )

    def _inference_cross_valid_init_scores(
        self,
        idl_gtvn_dir: str,
        dataset_ver: str,
        patients: List,
    ) -> Dict:

        scores = Dict()

        # init metrics of round 01
        for stats in [Stats.MEDIAN, Stats.AVG]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores[stats][metric]["round=01"] = List()

        # load baseline metrics on testset
        baseline_scores = g.load_json(
            os.path.join(
                Path(idl_gtvn_dir).parent,
                "baseline",
                "inference_{}_test.json".format(dataset_ver),
            )
        )

        # copy baseline gtvn scores of each patient
        for patient in patients:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores["patient={}".format(patient)][metric]["round=00"] = (
                    baseline_scores["patient={}".format(patient)]["gtvn"][metric]
                )

        # also copy baseline median and avg gtvn scores
        for stats in [Stats.MEDIAN, Stats.AVG]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores[stats][metric]["round=00"] = baseline_scores[stats]["gtvn"][
                    metric
                ]

        return scores

    def _inference_all_folds_save_patient_preds(
        self,
        patient: str,
        epoch_dir: str,
        patient_outputs: Dict,
    ):
        epoch_patient_dir = os.path.join(
            epoch_dir,
            "patients",
            "patient={}".format(patient),
        )
        g.create_dir(epoch_patient_dir)

        # create cross validation dir to save distance maps and clicks
        cross_valid_patient_dir = os.path.join(
            Path(epoch_dir).parent.parent,
            "patients",
            "patient={}".format(patient),
            "round=01",
        )
        g.create_dir(cross_valid_patient_dir)

        # save pred
        g.save_nii(
            img=patient_outputs["gtvn"]["pred"],
            save_path=os.path.join(epoch_patient_dir, "gtvn_pred.nii.gz"),
            spacing=g.NII_SPACING,
        )

        # save distance map and clicks
        for i in ["distance.map", "clicks"]:
            save_path = os.path.join(
                cross_valid_patient_dir,
                "gtvn_{}.nii.gz".format(i.replace(".", "_")),
            )
            if not os.path.exists(save_path):
                g.save_nii(
                    img=patient_outputs["gtvn"][i],
                    save_path=save_path,
                    spacing=g.NII_SPACING,
                )

    def _inference_all_folds_record_patient_score(
        self,
        patient: str,
        patient_outputs: Dict,
        scores: Dict,
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            # save cur patient metric
            scores["patient={}".format(patient)][metric] = patient_outputs["gtvn"][
                metric
            ]
            # add cur patient metric into a list for avg and median calculation
            # initialize a list
            if scores[Stats.AVG][metric] == {}:
                scores[Stats.AVG][metric] = List()
            # add current patient metric into the list
            scores[Stats.AVG][metric].append(patient_outputs["gtvn"][metric])

    def _inference_calculate_avg_median_save_json(
        self,
        scores: Dict,
        save_dir: str,
        dataset_ver: str,
        dataset_part: str,
        mda_obs: str = None,
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:

            # for validation set, no baseline metric, no need to add key "round=01"
            if dataset_part == DatasetPart.VALID:
                scores[Stats.MEDIAN][metric] = g.calculate_median(
                    scores[Stats.AVG][metric]
                )
                scores[Stats.AVG][metric] = g.calculate_avg(scores[Stats.AVG][metric])

            # for test set, add key "round=01" to compare with baseline metric
            elif dataset_part == DatasetPart.TEST:
                scores[Stats.MEDIAN][metric]["round=01"] = g.calculate_median(
                    scores[Stats.AVG][metric]["round=01"]
                )
                scores[Stats.AVG][metric]["round=01"] = g.calculate_avg(
                    scores[Stats.AVG][metric]["round=01"]
                )

        # save scores in json
        if mda_obs is None:
            json_name = "inference_{}_{}.json".format(dataset_ver, dataset_part)
        else:
            json_name = "inference_{}_{}_{}.json".format(
                dataset_ver, dataset_part, mda_obs
            )
        g.save_json(
            data=scores,
            path=os.path.join(save_dir, json_name),
        )

    def __is_valid_idl_gtvn_id(self, idl_gtvn_id: str):
        if not idl_gtvn_id.startswith("idl.gtvn_"):
            g.error_exit("'idl_gtvn_id' must start with 'idl.gtvn_'!")

    def _remove_non_optimal_epochs_record_epoch_scores(
        self,
        fold_scores: Dict,
        epoch_scores: Dict,
        epoch: str,
    ):

        for stats in [Stats.MEDIAN, Stats.AVG]:
            fold_scores[epoch][stats] = epoch_scores[stats]

    def _remove_non_optimal_epochs_find_best_epoch(
        self, scores: Dict, gtv_list: list = ["gtvn"]
    ):
        return super()._remove_non_optimal_epochs_find_best_epoch(
            scores=scores, gtv_list=gtv_list
        )

    def inference_cross_valid(
        self,
        idl_gtvn_id: str,
        dataset_ver: str = None,  # au/mda
        mda_obs: str = None,
        device_id: int = 1,  # use card 1 by default
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._inference_cross_valid(
            train_id=idl_gtvn_id,
            dataset_ver=dataset_ver,
            mda_obs=mda_obs,
            device_id=device_id,
            debug_mode=debug_mode,
        )

    def _inference_cross_valid_record_patient_score(
        self,
        patient: str,
        preds: Dict,
        labels: Dict,
        metric_funcs: Dict,
        scores: Dict,
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            score = metric_funcs[metric](preds["gtvn"], labels["gtvn"])
            # record current score
            scores["patient={}".format(patient)][metric]["round=01"] = score
            # record scores for avg and median score calculation
            if scores[Stats.AVG][metric]["round=01"] == {}:
                scores[Stats.AVG][metric]["round=01"] = List()
            scores[Stats.AVG][metric]["round=01"].append(score)

    def _inference_single_patient_load_dataset(
        self,
        patient: str,
        dataset_ver: str,
        no_pt: bool,
        no_mr: bool,
        idl_gtvn_geodesic_distance: bool,
        obs_gtvn_clicks: ndarray,
    ):
        return IDLGTVnDataSet(
            patients=[patient],
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            no_mr=no_mr,
            geodesic_distance=idl_gtvn_geodesic_distance,
            augment=None,
            obs_gtvn_clicks=obs_gtvn_clicks,
            random_click=False,
        )

    def _inference_single_patient_record_labels(
        self,
        outputs: Dict,
        dataset_item: Dict,
    ):
        labels = dataset_item["labels"][1].cpu().numpy()
        img_shape = dataset_item["shape"]
        labels = g.center_align_img(labels, img_shape)

        outputs["gtvn"]["label"] = labels

        return outputs

    def _inference_single_patient_record_gtvn_clicks(
        self,
        outputs: Dict,
        dataset_item: Dict,
    ):
        idl_gtvn_clicks = dataset_item["clicks"]
        img_shape = dataset_item["shape"]
        idl_gtvn_clicks = torch.squeeze(idl_gtvn_clicks, dim=0).cpu().numpy()
        idl_gtvn_clicks = g.center_align_img(idl_gtvn_clicks, img_shape)

        outputs["gtvn"]["clicks"] = idl_gtvn_clicks

    def _inference_single_patient_record_preds(
        self,
        outputs: Dict,
        preds: ndarray,
        img_shape: tuple,
    ):
        # preds: [background, gtvn]
        outputs["gtvn"]["pred"] = g.center_align_img(preds[1], img_shape)

    def _inference_single_patient_record_gtvn_distance_map(
        self,
        outputs: Dict,
        input_imgs: Tensor,
        img_shape: tuple,
    ):
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        input_imgs = torch.squeeze(input_imgs, dim=0).cpu().numpy()
        # input_imgs: [distance.map, "CT", "PT", "T1dr", "T2dr"]
        outputs["gtvn"]["distance.map"] = g.center_align_img(input_imgs[0], img_shape)

    def _inference_single_patient_gtvn_post_process(self, outputs: Dict):
        if 0:
            cc_list = g.get_connected_components(outputs["gtvn"]["pred"])
            outputs["gtvn"]["pred"] = np.zeros_like(outputs["gtvn"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * outputs["gtvn"]["clicks"]).sum() > 0:
                    outputs["gtvn"]["pred"] = np.maximum(
                        outputs["gtvn"]["pred"], cur_cc
                    )

    def _find_best_cnn_in_folds(self, idl_gtvn_id: str) -> str:
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)

        scores = Dict()

        idl_gtvn_dir = self._find_train_dir(idl_gtvn_id)

        fold_dirs = g.get_sub_dirs(
            input_dir=idl_gtvn_dir,
            key_word="fold=",
            full_path=True,
        )
        for fold_dir in fold_dirs:
            dataset_ver = g.load_json(os.path.join(fold_dir, "hyper.json"))[
                "dataset.ver"
            ]

            fold = Path(fold_dir).name
            epoch_dir = g.get_sub_dirs(fold_dir, key_word="epoch=", full_path=True)[0]
            epoch_scores = g.load_json(
                os.path.join(
                    epoch_dir,
                    "inference_{}_valid.json".format(dataset_ver),
                )
            )
            for stats in [Stats.MEDIAN, Stats.AVG]:
                scores[fold][stats] = epoch_scores[stats]

        for stats in [Stats.MEDIAN, Stats.AVG]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                # create a tmp list to sort
                list_to_sort = List()
                # add elements into the list
                for epoch in scores.keys():
                    list_to_sort.append(scores[epoch][stats][metric])
                # sort the list
                if metric == Metric.DSC:
                    list_to_sort.sort(reverse=False)
                else:
                    list_to_sort.sort(reverse=True)
                # update value based on the idx in the list
                for epoch in scores.keys():
                    new_value = list_to_sort.index(scores[epoch][stats][metric])
                    # if metric == Metric.DSC:
                    #     new_value *= 2
                    scores[epoch][stats][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stats in [Stats.AVG, Stats.MEDIAN]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    evaluation[epoch] += scores[epoch][stats][metric]

        best_fold = evaluation.key_with_max_value()
        best_epoch_dir = g.get_sub_dirs(
            os.path.join(idl_gtvn_dir, best_fold), key_word="epoch=", full_path=True
        )[0]
        best_cnn_path = g.get_sub_files(best_epoch_dir, key_word=".pt", full_path=True)[
            0
        ]
        return best_cnn_path

    def _is_valid_idl_dataset_ver(
        self,
        hyper: Dict,
        baseline_dataset_ver: str,
    ):
        if baseline_dataset_ver not in [
            DatasetVer.AU,
            DatasetVer.MDA,
            DatasetVer.NKI,
            DatasetVer.HECKTOR,
        ]:
            g.error_exit(ErrMsg.DATASET_VER_INVALID)

        if hyper["dataset.ver"] != baseline_dataset_ver:
            g.error_exit(ErrMsg.DATASET_VER_INVALID)
