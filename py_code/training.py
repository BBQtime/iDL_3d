import global_elems as g
import os
import torch
import math
import random
import torch.nn as nn
import numpy as np
from numpy import ndarray
from segment_metrics import SegmentationMetrics
from itertools import product
from collections import OrderedDict
from torch import optim
from idl_gtvt_dataset import IDLGTVtDataSet
from typing import Union
from unet_pp_slim import UNetPPSlim
from datetime import datetime
from baseline_dataset import BaselineDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom import Json
from custom import Dict
from custom import List
from custom import set_range


class Training:
    def __init__(self):
        # segmentation metrics
        self._metrics = Dict()
        for metric in g.METRICS:
            self._metrics[metric] = SegmentationMetrics(metric).to(g.DEVICE)

    # new hyper are loaded from group of new json files
    # baseline hyper are loaded from exist json file together with exist cnn
    # baseline hyper (cnn/dataset_pct/dataset_seed) only used for iDL
    def _load_hyper(self, hyper: Dict, cnn_path: str = "") -> None:

        # device name
        if torch.cuda.device_count() < 1:
            hyper["device"] = "cpu"
        else:
            hyper["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]

        # dropout
        hyper["dropout"] = set_range(hyper["dropout"], (0.0, 1.0))
        # batch size
        hyper["batch.size"] = g.MAX_BATCH_SIZE_PER_GPU
        if g.used_gpu_count() > 1:
            hyper["batch.size"] *= g.used_gpu_count()

        # = 1 will cause error
        hyper["lr"]["decay.factor"] = set_range(
            hyper["lr"]["decay.factor"], (g.EPS, 1 - g.EPS)
        )

        # augment methods
        hyper["augment"]["methods"] = List(hyper["augment"]["methods"])

        # augment lower/upper limit
        hyper["augment"]["up.limit"] = set_range(
            hyper["augment"]["up.limit"], (1, len(hyper["augment"]["methods"]))
        )
        hyper["augment"]["low.limit"] = set_range(
            hyper["augment"]["low.limit"], (1, hyper["augment"]["up.limit"])
        )

        # loss function parameters
        hyper["loss"]["weight"] = set_range(hyper["loss"]["weight"], (0.0, 1.0))
        hyper["loss"]["delta"] = set_range(hyper["loss"]["delta"], (0.0, 1.0))

        # load cnn
        self._load_cnn(hyper=hyper, cnn_path=cnn_path)

        # optimizer (no need to move to cuda)
        if isinstance(hyper["lr"]["actual"], list):
            lr = hyper["lr"]["actual"][0]
        else:
            lr = hyper["lr"]["actual"]
        hyper["optim"] = optim.Adam(params=hyper["cnn"].parameters(), lr=lr)

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

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_cnn(self, hyper: Dict, cnn_path: str = ""):
        # new model
        if cnn_path == "" or cnn_path is None:
            hyper["cnn"] = UNetPPSlim(
                in_chan=4, out_chan=3, dropout=hyper["dropout"]
            ).to(g.DEVICE)

        # exist cnn
        else:
            # load state dict only
            if g.CNN_STATE_DICT_ONLY:
                hyper["cnn"] = UNetPPSlim(
                    in_chan=4, out_chan=3, dropout=hyper["dropout"]
                ).to(g.DEVICE)
                hyper["cnn"].load_state_dict(torch.load(cnn_path))

            # load entire cnn
            else:
                hyper["cnn"] = torch.load(cnn_path).to(g.DEVICE)

        # set multi-GPU
        if g.used_gpu_count() > 1:
            hyper["cnn"] = nn.DataParallel(hyper["cnn"]).to(g.DEVICE)

    def __get_simple_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for i in hyper:
            if i == "metrics":
                pass

            elif i == "augment":
                simple_hyper[i] = hyper[i].copy()
                simple_hyper[i]["methods"] = simple_hyper[i]["methods"].to_str()

            elif i == "loss":
                simple_hyper[i] = hyper[i].copy()
                simple_hyper[i]["func"] = "unified.focal.loss"

            elif i == "cnn":
                simple_hyper[i] = "unet.pp.slim"

            elif i == "optim":
                simple_hyper[i] = "adam"

            elif i == "scheduler":
                simple_hyper[i] = "reduce.lr.on.plateau"

            elif isinstance(hyper[i], list) or isinstance(hyper[i], dict):
                simple_hyper[i] = hyper[i].copy()
            else:
                simple_hyper[i] = hyper[i]

        simple_hyper = dict(OrderedDict(sorted(simple_hyper.items())))
        for i in simple_hyper:
            if isinstance(simple_hyper[i], dict):
                simple_hyper[i] = dict(OrderedDict(sorted(simple_hyper[i].items())))

        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self.__get_simple_hyper(hyper)
        for key, value in simple_hyper.items():
            print(key + ":", value)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        Json.save(data=simple_hyper, path=json_path)

    # split dataset and save result into json file
    def _split_dataset(self) -> Dict:
        dataset_split = Json.load(g.DATASET_SPLIT_JSON)
        train_patients = List()

        for fold in dataset_split.keys():
            if fold != "test.set":
                train_patients += List(dataset_split[fold])
                dataset_split.pop(fold)

        random.shuffle(train_patients)

        fold_len = round(len(train_patients) / g.DATASET_K_FOLDS)
        for fold in range(1, g.DATASET_K_FOLDS + 1):
            if fold == g.DATASET_K_FOLDS:
                dataset_split["fold.{}".format(fold)] = train_patients.to_str()
            else:
                dataset_split["fold.{}".format(fold)] = train_patients[
                    :fold_len
                ].to_str()
                train_patients = train_patients[fold_len:]

        Json.save(dataset_split, g.DATASET_SPLIT_JSON)
        return dataset_split

    def _save_cnn(self, hyper: Dict, save_path: str):
        if not save_path.endswith(".pt"):
            save_path += ".pt"

        # save state dict only
        if g.CNN_STATE_DICT_ONLY:
            if g.used_gpu_count() > 1:
                torch.save(hyper["cnn"].module.state_dict(), save_path)
            else:
                torch.save(hyper["cnn"].state_dict(), save_path)

        # save entire cnn
        else:
            if g.used_gpu_count() > 1:
                torch.save(hyper["cnn"].module, save_path)
            else:
                torch.save(hyper["cnn"], save_path)

    # train_id = start_time + train_remark
    def _init_train_id(
        self,
        train_remark: str,
        hyper_json_path: str,
        hyper: Dict,
        debug_mode: bool,
    ) -> str:

        train_id = self._get_cur_time_str()

        if debug_mode:
            train_id += "_delete.this"

        if train_remark != "" and train_remark is not None:
            while train_remark.startswith("_"):
                train_remark = train_remark[1:]
            while train_remark.endswith("_"):
                train_remark = train_remark[:-1]
            train_id += "_" + train_remark

        # add important hyper param (that need to be compared) to train_id
        origin_hyper_dict = Json.load(hyper_json_path)
        # make sure all values of hyper dict are "list" type
        for i in origin_hyper_dict:
            if not isinstance(origin_hyper_dict[i], list):
                origin_hyper_dict[i] = [origin_hyper_dict[i]]

        for key in origin_hyper_dict:
            if len(origin_hyper_dict[key]) > 1:
                # replace "_" with "." in key name
                train_id += "_" + key.replace("_", ".")
                train_id += "=" + str(hyper[key]).replace("_", ".").replace(" ", "")

        return train_id

    def _optimize_batch_size(
        self, hyper: Dict, dataset: Union[BaselineDataSet, IDLGTVtDataSet]
    ):
        dataset_len = dataset.__len__()
        if dataset_len > hyper["batch.size"]:
            hyper["batch.size"] = math.ceil(
                dataset_len / (math.ceil(dataset_len / hyper["batch.size"]))
            )
        else:
            hyper["batch.size"] = dataset_len

    def _inference_single_patient(
        self,
        patient: str,
        hyper: Dict,
        gtvt_only: bool,
        unetpp_output: int = 3,
        masked_label: ndarray = None,
    ) -> Dict:

        # result structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
        result = Dict()

        dataset = BaselineDataSet(patient_list=[patient])
        origin = Dict()

        # load gtvt
        origin["gtvt"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )

        if not gtvt_only:
            # load gtvs
            origin["gtvs"] = g.load_nii(
                os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)),
                binary=True,
            )
            # load gtvn
            gtvn_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(patient)
            )
            if os.path.exists(gtvn_path):
                origin["gtvn"] = g.load_nii(gtvn_path, binary=True)
            else:
                origin["gtvn"] = origin["gtvs"] - origin["gtvt"]

        hyper["cnn"].eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            inputs, labels = dataset.get_item(patient=patient)
            inputs = torch.unsqueeze(inputs.to(g.DEVICE), dim=0)
            labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
            outputs = hyper["cnn"].forward(inputs)[unetpp_output]
            # squeeze batch
            outputs = torch.squeeze(outputs, dim=0).cpu().numpy()

        # get gtvt/gtvn/gtvs img
        result["gtvt"]["pred"] = outputs[1]
        if not gtvt_only:
            result["gtvn"]["pred"] = outputs[2]
            result["gtvs"]["pred"] = np.maximum(outputs[1], outputs[2])

        # pad and crop to original size
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            if gtv == "gtvt" or not gtvt_only:
                result[gtv]["pred"] = g.central_pad(
                    result[gtv]["pred"], origin[gtv].shape
                )
                result[gtv]["pred"] = g.central_crop(
                    result[gtv]["pred"], origin[gtv].shape
                )

        # idl post processing
        if masked_label is not None and gtvt_only:
            cc_list = g.get_connected_components(result["gtvt"]["pred"])
            result["gtvt"]["pred"] = np.zeros_like(result["gtvt"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * masked_label).sum() > 0:
                    result["gtvt"]["pred"] = np.maximum(result["gtvt"]["pred"], cur_cc)

        # calculate segment scores
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            if gtv == "gtvt" or not gtvt_only:
                for metric in g.METRICS:
                    result[gtv][metric] = self._metrics[metric](
                        result[gtv]["pred"], origin[gtv]
                    )
        return result

    # protected function
    def _get_cur_time_str(self) -> str:
        start_time = str(datetime.now().replace(microsecond=0))
        start_time = start_time.replace(":", ".")
        start_time = start_time.replace("-", ".")
        start_time = start_time.replace(" ", ".")
        return start_time

    def _load_group_hyper(self, json_path: str) -> dict:
        group_hyper = List()
        origin_hyper_dict = Json.load(json_path)
        hyper_keys = origin_hyper_dict.keys()

        # make sure all values of hyper dict are "list" type
        for i in origin_hyper_dict:
            if not isinstance(origin_hyper_dict[i], list):
                origin_hyper_dict[i] = [origin_hyper_dict[i]]

        # get all cartesian products of hyper dict values
        for cur_values in product(*origin_hyper_dict.values()):
            # create current hyper param combination
            cur_hyper = Dict()
            for i in range(len(cur_values)):
                # copy the value if it is a dict, avoid using same memory
                if isinstance(cur_values[i], dict):
                    cur_hyper[hyper_keys[i]] = cur_values[i].copy()
                else:
                    cur_hyper[hyper_keys[i]] = cur_values[i]
            group_hyper.append(cur_hyper)

        return group_hyper
