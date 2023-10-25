import os
from pathlib import Path

import numpy as np
import torch
from custom import DatasetPart, Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Metric, Nii, Value
from dataset_idl_gtvn import DataSetIDLGTVn
from loss_func_idl_gtvn import UnifiedFocalLossIDLGTVn
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm
from training_baseline import TrainingBaseline


class TrainingIDLGTVn(TrainingBaseline):
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
                augment["methods"] = hyper["augment.methods"]
                augment["pct"] = hyper["augment.pct"]
                augment["min"] = hyper["augment.min"]
                augment["max"] = hyper["augment.max"]
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

    def real_idl(
        self,
        idl_gtvn_id: str,
        patient: str,
        idl_gtvn_clicks: ndarray,
        dataset_part: str,  # train/valid/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
    ):
        print("")
        print("real idl: {}".format(idl_gtvn_id))

        baseline_id = "baseline_real.idl"
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)

        cnn_idl_gtvn_dir = os.path.join(baseline_dir, "idl.gtvn_real.idl")
        if not os.path.exists(cnn_idl_gtvn_dir):
            Debug.error_exit("'idl.gtvn_real.idl' folder not found")

        output_idl_gtvn_dir = os.path.join(baseline_dir, idl_gtvn_id)
        if not os.path.exists(output_idl_gtvn_dir):
            Dir.create(output_idl_gtvn_dir)

        cnn_fold_dirs = Dir.get_sub_dirs(
            cnn_idl_gtvn_dir, key_word="fold=", full_path=True
        )
        hyper = Json.load(os.path.join(cnn_fold_dirs[0], "hyper.json"))
        no_pt = hyper["no.pt"]
        training_dataset_ver = hyper["dataset.ver"]

        dataset_ver = self._is_valid_dataset_version(
            dataset_ver=dataset_ver,
            origin_dataset_ver=training_dataset_ver,
        )
        self._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )
        print("dataset version: {}".format(dataset_ver))
        print("dataset section: {}".format(dataset_part))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics(dataset_ver)

        # loop through fold dirs
        for cnn_fold_dir in tqdm(cnn_fold_dirs):
            # fold = int(Path(cnn_fold_dir).name[len("fold=") :])
            # print("")
            # print("fold: ", fold)

            output_fold_dir = os.path.join(output_idl_gtvn_dir, Path(cnn_fold_dir).name)
            Dir.create(output_fold_dir)

            # loop through epoch dirs
            for cnn_epoch_dir in Dir.get_sub_dirs(
                cnn_fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = int(Path(cnn_epoch_dir).name[len("epoch=") :])
                # print("epoch: ", epoch)

                output_epoch_dir = os.path.join(
                    output_fold_dir, Path(cnn_epoch_dir).name
                )
                Dir.create(output_epoch_dir)

                # load cnn
                cnn_path = os.path.join(cnn_epoch_dir, "epoch={:03d}.pt".format(epoch))
                cnn = self._load_exist_cnn(cnn_path)

                # initialize scores dict
                patients = Dict()
                patients[dataset_part] = [patient]
                epoch_scores = self._inference_init_scores(
                    baseline_id=baseline_id,
                    dataset_ver=dataset_ver,
                    dataset_part=dataset_part,
                    patients=patients,
                )

                # outputs structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                patient_outputs = self._inference_single_patient(
                    patient=patient,
                    cnn=cnn,
                    dataset_ver=dataset_ver,
                    dataset_part=dataset_part,
                    no_pt=no_pt,
                    segment_metrics=segment_metrics,
                    idl_gtvn_baseline_id=baseline_id,
                    idl_gtvn_clicks=idl_gtvn_clicks,
                )

                # create folder and save preds of current patient
                self._fold_wise_inference_save_patient_preds(
                    patient=patient,
                    epoch_dir=output_epoch_dir,
                    patient_outputs=patient_outputs,
                    dataset_ver=dataset_ver,
                    dataset_part=dataset_part,
                )

                # record score of current patient (test and valid sets only)
                self._fold_wise_inference_record_patient_score(
                    patient=patient,
                    patient_outputs=patient_outputs,
                    scores=epoch_scores,
                )

                # all patients under current epoch have been traversed
                # calculate median and avg score of current epoch
                self._inference_calculate_save_avg_median(
                    scores=epoch_scores,
                    save_dir=output_epoch_dir,
                    dataset_ver=dataset_ver,
                    dataset_part=dataset_part,
                )

                continue  # next epoch

        # cross valid
        # initialize scores dict
        patients = Dict()
        patients[dataset_part] = [patient]
        cross_valid_scores = self._inference_init_scores(
            baseline_id=baseline_id,
            dataset_ver=dataset_ver,
            dataset_part=dataset_part,
            patients=patients,
        )

        # initialize preds
        preds = Dict()
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            preds[gtv] = None

        output_fold_dirs = Dir.get_sub_dirs(
            output_idl_gtvn_dir, key_word="fold=", full_path=True
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
            output_idl_gtvn_dir, "patients", "patient={}".format(patient)
        )
        pred_dir = os.path.join(pred_dir, "round=01")
        Dir.create(pred_dir)

        # save cross_valid preds (only save gtvt and gtvn)
        for gtv in preds.keys():
            if gtv != "gtvs":
                Nii.save(
                    img=preds[gtv],
                    save_path=os.path.join(pred_dir, "{}_pred.nii.gz".format(gtv)),
                    spacing=g.NII_SPACING[dataset_ver],
                )

        # load labels and calculate metrics
        labels = Img.load_labels(
            dataset_dir=g.DATASET_DIR[dataset_ver], patient=patient
        )
        self._cross_valid_inference_record_patient_score(
            patient=patient,
            preds=preds,
            labels=labels,
            segment_metrics=segment_metrics,
            scores=cross_valid_scores,
        )

        # calculate avg and median score
        self._inference_calculate_save_avg_median(
            scores=cross_valid_scores,
            save_dir=output_idl_gtvn_dir,
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )

        print("")
        print("real idl done!")

    def fold_wise_inference(
        self,
        idl_gtvn_id: str,
        dataset_part: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._fold_wise_inference(
            train_id=idl_gtvn_id,
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _inference_init_scores(
        self, baseline_id: str, dataset_ver: str, dataset_part: str, patients: Dict
    ) -> Dict:
        scores = Dict()

        # init round 01
        for stats in ["median", "avg"]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                scores[stats][metric]["round=01"] = List()

        # only load baseline scores of test set, because there is no validation scores
        if DatasetPart.TEST in dataset_part:
            # load baseline scores
            baseline_scores = Json.load(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    "baseline",
                    "inference_{}_{}.json".format(dataset_ver, dataset_part),
                )
            )

            # copy baseline gtvn scores of each patient
            for patient in patients[dataset_part]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    scores["patient={}".format(patient)][metric][
                        "round=00"
                    ] = baseline_scores["patient={}".format(patient)]["gtvn"][metric]

            # also copy baseline median and avg gtvn scores
            for stats in ["median", "avg"]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    scores[stats][metric]["round=00"] = baseline_scores[stats]["gtvn"][
                        metric
                    ]

        return scores

    def _fold_wise_inference_save_patient_preds(
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
            spacing=g.NII_SPACING[dataset_ver],
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
                    spacing=g.NII_SPACING[dataset_ver],
                )

    def _fold_wise_inference_record_patient_score(
        self, patient: str, patient_outputs: Dict, scores: Dict
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            # save cur patient score
            scores["patient={}".format(patient)][metric]["round=01"] = patient_outputs[
                "gtvn"
            ][metric]
            # add scores of current patient into median(list)
            for stats in ["median", "avg"]:
                scores[stats][metric]["round=01"].append(
                    patient_outputs["gtvn"][metric]
                )

    def _inference_calculate_save_avg_median(
        self, scores: Dict, save_dir: str, dataset_ver: str, dataset_part: str
    ):
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            scores["median"][metric]["round=01"] = Value.median(
                scores["median"][metric]["round=01"]
            )
            scores["avg"][metric]["round=01"] = Value.avg(
                scores["avg"][metric]["round=01"]
            )

        # save scores in json
        Json.save(
            data=scores,
            path=os.path.join(
                save_dir, "inference_{}_{}.json".format(dataset_ver, dataset_part)
            ),
        )

    def __is_valid_idl_gtvn_id(self, idl_gtvn_id: str):
        if not idl_gtvn_id.startswith("idl.gtvn_"):
            Debug.error_exit("idl.gtvn id error")

    def remove_non_optimal_epochs(self, idl_gtvn_id: str):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._remove_non_optimal_epochs(idl_gtvn_id)

    def _remove_non_optimal_epochs_record_epoch_scores(
        self, fold_scores: Dict, epoch_scores: Dict, epoch: str
    ):
        for stats in ["median", "avg"]:
            for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                fold_scores[epoch][stats][metric] = epoch_scores[stats][metric][
                    "round=01"
                ]

    def _remove_non_optimal_epochs_find_best_epoch(
        self, scores: Dict, gtv_list: list = ["gtvn"]
    ):
        return super()._remove_non_optimal_epochs_find_best_epoch(
            scores=scores, gtv_list=gtv_list
        )

    def cross_valid_inference(
        self,
        idl_gtvn_id: str,
        dataset_part: str,  # test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._cross_valid_inference(
            train_id=idl_gtvn_id,
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _cross_valid_inference_is_valid_dataset_part(
        self,
        dataset_part: str,
        dataset_ver: str,
    ):
        if dataset_part not in [
            DatasetPart.TEST,
            DatasetPart.TEST_INTER,
            DatasetPart.TEST_EXTER,
        ]:
            Debug.error_exit(
                "'dataset_part' can not take on any values other than 'test/test.inter/test.exter'"
            )

        self._is_valid_dataset_part(
            dataset_part=dataset_part,
            dataset_ver=dataset_ver,
        )

    def _cross_valid_inference_record_patient_score(
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
            for stats in ["median", "avg"]:
                scores[stats][metric]["round=01"].append(score)

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
