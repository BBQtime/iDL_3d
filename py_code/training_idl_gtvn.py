import os
from pathlib import Path

import numpy as np
import torch
from custom import Debug, Dict, Directory, Folder
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from dataset_idl_gtvn import DataSetIDLGTVn
from loss_func_idl_gtvn import UnifiedFocalLossIDLGTVn
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
        for i in ["train", "valid"]:
            # only use data augmentation on training set
            if i == "train":
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

    def fold_wise_inference(
        self,
        idl_gtvn_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._fold_wise_inference(
            train_id=idl_gtvn_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _inference_init_scores(
        self, baseline_id: str, dataset_ver: str, dataset_section: str, patients: Dict
    ) -> Dict:
        scores = Dict()

        # init round 01
        for stats in ["median", "avg"]:
            for metric in g.METRICS:
                scores[stats][metric]["round=01"] = List()

        # only load baseline scores of test set, because there is no validation scores
        if "test" in dataset_section:
            # load baseline scores
            baseline_scores = Json.load(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    baseline_id,
                    "baseline",
                    "inference_{}_{}.json".format(dataset_ver, dataset_section),
                )
            )

            # copy baseline gtvn scores of each patient
            for patient in patients[dataset_section]:
                for metric in g.METRICS:
                    scores["patient={}".format(patient)][metric][
                        "round=00"
                    ] = baseline_scores["patient={}".format(patient)]["gtvn"][metric]

            # also copy baseline median and avg gtvn scores
            for stats in ["median", "avg"]:
                for metric in g.METRICS:
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
        dataset_section: str,
    ):
        if dataset_section == "train" or dataset_section == "valid":
            return

        epoch_patient_dir = os.path.join(
            epoch_dir,
            "patients",
            "patient={}".format(patient),
        )
        Folder.create(epoch_patient_dir)

        # create cross validation dir to save distance maps and clicks
        cross_valid_patient_dir = os.path.join(
            Path(epoch_dir).parent.parent,
            "patients",
            "patient={}".format(patient),
            "round=01",
        )
        Folder.create(cross_valid_patient_dir)

        # save pred
        Nii.save(
            img=patient_outputs["gtvn"]["pred"],
            save_path=os.path.join(epoch_patient_dir, "gtvn_pred.nii"),
            spacing=g.NII_SPACING[dataset_ver],
        )
        # save distance map and clicks
        for i in ["distance.map", "clicks"]:
            save_path = os.path.join(
                cross_valid_patient_dir,
                "gtvn_{}.nii".format(i.replace(".", "_")),
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
        for metric in g.METRICS:
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
        self, scores: Dict, save_dir: str, dataset_ver: str, dataset_section: str
    ):
        for metric in g.METRICS:
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
                save_dir, "inference_{}_{}.json".format(dataset_ver, dataset_section)
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
            for metric in g.METRICS:
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
        dataset_section: str,  # test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self.__is_valid_idl_gtvn_id(idl_gtvn_id)
        self._cross_valid_inference(
            train_id=idl_gtvn_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _cross_valid_inference_is_valid_dataset_section(
        self,
        dataset_section: str,
        dataset_ver: str,
    ):
        if dataset_section not in ["test", "test.inter", "test.exter"]:
            Debug.error_exit(
                "'dataset_section' can not take on any values other than 'test/test.inter/test.exter'"
            )

        self._is_valid_dataset_section(
            dataset_section=dataset_section,
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
        for metric in g.METRICS:
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
    ):
        return DataSetIDLGTVn(
            patients=[patient],
            baseline_id=idl_gtvn_baseline_id,
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            augment=None,
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
