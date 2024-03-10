import os
from pathlib import Path

import numpy as np
import torch
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from dataset_idl_gtvn import DataSetIDLGTVn
from loss_func_idl_gtvn import UnifiedFocalLossIDLGTVn
from numpy import ndarray
from PyQt5.QtCore import pyqtSignal
from str_lib import DatasetPart, DatasetVer, Metric, Stat
from torch import Tensor
from training_baseline import TrainingBaseline
from training_core import ObsStudyProgress


class ObsStudyGTVnProgress(ObsStudyProgress):
    class ProgressStep:
        INFERENCE_INIT = 1
        INFERENCE_LOAD_IMG = 3
        INFERENCE_FORWARD = 1
        INFERENCE_SAVE_PRED = 1
        CROSS_VALID = 3

    def __init__(self):
        super().__init__()
        self.step = self.ProgressStep()


class TrainingIDLGTVn(TrainingBaseline):
    def __init__(self, idl_progress_signal: pyqtSignal = None):
        super().__init__()
        if idl_progress_signal is not None:
            self._obs_study_progress = ObsStudyGTVnProgress()
            self._obs_study_progress.progress_signal = idl_progress_signal
        else:
            self._obs_study_progress = None

    def _load_hyper_new_cnn(self, hyper: Dict, in_chan: int = 5, out_chan: int = 2):
        # no need to reduce cnn input channel here if no_pt=true
        # it will be reduced in super()._load_hyper_new_cnn
        super()._load_hyper_new_cnn(hyper=hyper, in_chan=in_chan, out_chan=out_chan)

    def _load_hyper_loss_func(self, hyper: Dict):
        hyper["loss.func"] = UnifiedFocalLossIDLGTVn(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

    def _load_hyper_data_sets(self, hyper: Dict, idl_gtvn_baseline_id: str):
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
            hyper["{}.set".format(i)] = DataSetIDLGTVn(
                patients=hyper["{}.patients".format(i)],
                baseline_id=idl_gtvn_baseline_id,
                dataset_ver=hyper["dataset.ver"],
                no_pt=hyper["no.pt"],
                augment=augment,
                random_click=False,
            )

    def new_training(
        self, baseline_id: str, train_remark: str = "", debug_mode: bool = False
    ):
        self._is_valid_baseline_id(baseline_id)
        self._new_training(
            idl_gtvn_baseline_id=baseline_id,
            train_remark=train_remark,
            debug_mode=debug_mode,
        )

    def obs_study(
        self,
        idl_gtvn_id: str,
        patient: str,
        idl_gtvn_clicks: ndarray = None,  # None means no gtvn click
        debug_mode=False,
    ):
        print("")
        print("observer study: {}".format(idl_gtvn_id))

        baseline_id = "baseline_obs.study"
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)

        obs_study_cnns_base_dir = os.path.join(baseline_dir, "idl.gtvn_obs.study")
        if not os.path.exists(obs_study_cnns_base_dir):
            Debug.error_exit("'idl.gtvn_obs.study' folder not found!")

        obs_study_output_dir = os.path.join(baseline_dir, idl_gtvn_id)
        if not os.path.exists(obs_study_output_dir):
            Dir.create(obs_study_output_dir)

        cnn_fold_dirs = Dir.get_sub_dirs(
            obs_study_cnns_base_dir, key_word="fold=", full_path=True
        )
        hyper = Json.load(os.path.join(cnn_fold_dirs[0], "hyper.json"))
        no_pt = hyper["no.pt"]

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics()

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

        # loop through fold dirs
        for cnn_fold_dir in cnn_fold_dirs:
            output_fold_dir = os.path.join(
                obs_study_output_dir, Path(cnn_fold_dir).name
            )
            Dir.create(output_fold_dir)

            cnn_epoch_dir = Dir.get_sub_dirs(
                cnn_fold_dir, key_word="epoch=", full_path=True
            )[0]
            epoch = int(Path(cnn_epoch_dir).name[len("epoch=") :])

            output_epoch_dir = os.path.join(output_fold_dir, Path(cnn_epoch_dir).name)
            Dir.create(output_epoch_dir)

            # load cnn
            cnn_path = os.path.join(cnn_epoch_dir, "epoch={:03d}.pt".format(epoch))
            cnn = self._load_exist_cnn(cnn_path)

            # idl progress INFERENCE_INIT
            # self._timer.cal_duration("INFERENCE_INIT")
            if self._obs_study_progress is not None:
                self._obs_study_progress.cur_step += (
                    self._obs_study_progress.step.INFERENCE_INIT
                )
                self._obs_study_progress.emit_signal()

            # outputs structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
            patient_outputs = self._inference_single_patient(
                patient=patient,
                cnn=cnn,
                dataset_ver=DatasetVer.OBS_STUDY,
                dataset_part=DatasetPart.TEST,
                no_pt=no_pt,
                segment_metrics=segment_metrics,
                idl_gtvn_baseline_id=baseline_id,
                idl_gtvn_clicks=idl_gtvn_clicks,
            )

            # create folder and save preds of current patient
            self._inference_on_folds_save_patient_preds(
                patient=patient,
                epoch_dir=output_epoch_dir,
                patient_outputs=patient_outputs,
                dataset_ver=DatasetVer.OBS_STUDY,
                dataset_part=DatasetPart.TEST,
            )

            # idl progress INFERENCE_SAVE_PRED
            # self._timer.cal_duration("INFERENCE_SAVE_PRED")
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

        output_fold_dirs = Dir.get_sub_dirs(
            obs_study_output_dir, key_word="fold=", full_path=True
        )
        for output_fold_dir in output_fold_dirs:
            # find epoch dir
            output_epoch_dir = Dir.get_sub_dirs(
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
                    img = Nii.load(path=pred_path, binary=False)
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
        Dir.create(pred_dir)

        # save cross_valid preds (only save gtvt and gtvn)
        for gtv in preds.keys():
            if gtv != "gtvs":
                Nii.save(
                    img=preds[gtv],
                    save_path=os.path.join(pred_dir, "{}_pred.nii.gz".format(gtv)),
                    spacing=g.NII_SPACING,
                )

        # idl progress CROSS_VALID
        # self._timer.cal_duration("CROSS_VALID")
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step += (
                self._obs_study_progress.step.CROSS_VALID
            )
            self._obs_study_progress.emit_signal()

        # if self._obs_study_progress is not None:
        #     print(self._obs_study_progress.cur_step, self._obs_study_progress.total_step)

    def inference_on_folds(
        self,
        idl_gtvn_id: str,
        dataset_part: str,  # train/test
        dataset_ver: str = None,  # au/mda
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._inference_on_folds(
            train_id=idl_gtvn_id,
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _inference_init_scores(
        self,
        baseline_id: str,
        dataset_ver: str,
        dataset_part: str,
        patients: Dict,
    ) -> Dict:
        scores = Dict()

        # init round 01
        for stat in [Stat.MEDIAN, Stat.AVG]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores[stat][metric]["round=01"] = List()

        # only load baseline scores of test set, because there is no validation scores
        if DatasetPart.TEST in dataset_part:
            # load baseline scores
            baseline_scores = Json.load(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    "baseline",
                    "inference_{}.json".format(dataset_ver),
                )
            )

            # copy baseline gtvn scores of each patient
            for patient in patients[dataset_part]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    scores["patient={}".format(patient)][metric]["round=00"] = (
                        baseline_scores["patient={}".format(patient)]["gtvn"][metric]
                    )

            # also copy baseline median and avg gtvn scores
            for stat in [Stat.MEDIAN, Stat.AVG]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    scores[stat][metric]["round=00"] = baseline_scores[stat]["gtvn"][
                        metric
                    ]

        return scores

    def _inference_on_folds_save_patient_preds(
        self,
        patient: str,
        epoch_dir: str,
        patient_outputs: Dict,
        dataset_ver: str,
        dataset_part: str,
    ):
        if dataset_part == DatasetPart.TRAIN or dataset_part == DatasetPart.VALID:
            return

        epoch_patient_dir = os.path.join(
            epoch_dir,
            "patients",
            "patient={}".format(patient),
        )
        Dir.create(epoch_patient_dir)

        # create cross validation dir to save distance maps and clicks
        cross_valid_patient_dir = os.path.join(
            Path(epoch_dir).parent.parent,
            "patients",
            "patient={}".format(patient),
            "round=01",
        )
        Dir.create(cross_valid_patient_dir)

        # save pred
        Nii.save(
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
                Nii.save(
                    img=patient_outputs["gtvn"][i],
                    save_path=save_path,
                    spacing=g.NII_SPACING,
                )

    def _inference_on_folds_record_patient_score(
        self, patient: str, patient_outputs: Dict, scores: Dict
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            # save cur patient score
            scores["patient={}".format(patient)][metric]["round=01"] = patient_outputs[
                "gtvn"
            ][metric]
            # add scores of current patient into median(list)
            for stat in [Stat.MEDIAN, Stat.AVG]:
                scores[stat][metric]["round=01"].append(patient_outputs["gtvn"][metric])

    def _inference_calculate_save_avg_median(
        self,
        scores: Dict,
        save_dir: str,
        dataset_ver: str,
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            scores[Stat.MEDIAN][metric]["round=01"] = Value.median(
                scores[Stat.MEDIAN][metric]["round=01"]
            )
            scores[Stat.AVG][metric]["round=01"] = Value.avg(
                scores[Stat.AVG][metric]["round=01"]
            )

        # save scores in json
        Json.save(
            data=scores,
            path=os.path.join(save_dir, "inference_{}.json".format(dataset_ver)),
        )

    def __is_valid_idl_gtvn_id(self, idl_gtvn_id: str):
        if not idl_gtvn_id.startswith("idl.gtvn_"):
            Debug.error_exit("'idl_gtvn_id' must start with 'idl.gtvn_'!")

    def remove_non_optimal_epochs(self, idl_gtvn_id: str):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._remove_non_optimal_epochs(idl_gtvn_id)

    def _remove_non_optimal_epochs_record_epoch_scores(
        self, fold_scores: Dict, epoch_scores: Dict, epoch: str
    ):
        for stat in [Stat.MEDIAN, Stat.AVG]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                fold_scores[epoch][stat][metric] = epoch_scores[stat][metric][
                    "round=01"
                ]

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
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._inference_cross_valid(
            train_id=idl_gtvn_id,
            dataset_part=DatasetPart.TEST,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _inference_cross_valid_record_patient_score(
        self,
        patient: str,
        preds: Dict,
        labels: Dict,
        segment_metrics: Dict,
        scores: Dict,
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            score = segment_metrics[metric](preds["gtvn"], labels["gtvn"])
            # record current score
            scores["patient={}".format(patient)][metric]["round=01"] = score
            # record scores for avg and median score calculation
            for stat in [Stat.MEDIAN, Stat.AVG]:
                scores[stat][metric]["round=01"].append(score)

    def _inference_single_patient_load_dataset(
        self,
        patient: str,
        dataset_ver: str,
        no_pt: bool,
        idl_gtvn_baseline_id: str,
        idl_gtvn_clicks: ndarray,
    ):
        return DataSetIDLGTVn(
            patients=[patient],
            baseline_id=idl_gtvn_baseline_id,
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            augment=None,
            gtvn_clicks=idl_gtvn_clicks,
            random_click=False,
        )

    def _inference_single_patient_record_labels(self, labels: Dict):
        outputs = Dict()
        outputs["gtvn"]["label"] = labels["gtvn"]
        return outputs

    def _inference_single_patient_get_gtvn_clicks(self, item: list):
        return item[2]

    def _inference_single_patient_record_outputs(
        self, outputs: Dict, preds: Dict, input_imgs: Tensor, idl_gtvn_clicks: Tensor
    ):
        outputs["gtvn"]["pred"] = preds[1]
        input_imgs = torch.squeeze(input_imgs, dim=0).cpu().numpy()
        outputs["gtvn"]["distance.map"] = input_imgs[0]
        outputs["gtvn"]["clicks"] = torch.squeeze(idl_gtvn_clicks, dim=0).cpu().numpy()

    def _inference_single_patient_gtvn_post_process(self, outputs: Dict):
        if 0:
            cc_list = Img.connected_components(outputs["gtvn"]["pred"])
            outputs["gtvn"]["pred"] = np.zeros_like(outputs["gtvn"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * outputs["gtvn"]["clicks"]).sum() > 0:
                    outputs["gtvn"]["pred"] = np.maximum(
                        outputs["gtvn"]["pred"], cur_cc
                    )
