import os
from pathlib import Path

import numpy as np
import torch
from custom import Debug, Dict, Directory, Folder
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from dataset_idl_gtvn import DataSetIDLGTVn
from loss_func_idl_gtvn import UnifiedFocalLossIDLGTVn
from tqdm import tqdm
from training_baseline import TrainingBaseline


class TrainingIDLGTVn(TrainingBaseline):
    def _load_hyper_new_cnn(self, hyper: Dict, in_chan: int = 5, out_chan: int = 2):
        super()._load_hyper_new_cnn(hyper=hyper, in_chan=in_chan, out_chan=out_chan)

    def _load_hyper_loss_func(self, hyper: Dict):
        hyper["loss.func"] = UnifiedFocalLossIDLGTVn(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

    def _load_hyper_data_sets(self, hyper: Dict):
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
                baseline_id=hyper["baseline.id"],
                dataset_ver=hyper["dataset.ver"],
                augment=augment,
                random_click=False,
            )

    def new_training(
        self, baseline_id: str, train_remark: str = "", debug_mode: bool = False
    ):
        if not baseline_id.startswith("baseline_"):
            Debug.error_exit("baseline id error")
        self._new_training(
            train_type="idl.gtvn",
            baseline_id=baseline_id,
            train_remark=train_remark,
            debug_mode=debug_mode,
        )

    def inference(
        self,
        idl_gtvn_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self._inference(
            inference_type="idl.gtvn",
            train_id=idl_gtvn_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def remove_non_optimal_epochs(self, idl_gtvn_id: str):
        self._remove_non_optimal_epochs(inference_type="idl.gtvn", train_id=idl_gtvn_id)

    def cross_valid_evaluation(
        self,
        idl_gtvn_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self._cross_valid_evaluation(
            inference_type="idl.gtvn",
            train_id=idl_gtvn_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )
