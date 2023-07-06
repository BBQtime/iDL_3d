from custom import Global as g
import os
import torch
import math
import random
import numpy as np
from pathlib import Path
from torch.nn import DataParallel
from torch.nn import ModuleDict
from numpy import ndarray
from itertools import product
from collections import OrderedDict
from torch import optim
from unet_pp_slim import UNetPPSlim
from unet_slim import UNetSlim
from datetime import datetime
from metrics_lib import SegmentationMetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom import Json
from custom import Dict
from custom import List
from custom import GPU
from custom import ValueUtils
from custom import Explorer
from custom import Debug


class Training:
    def _load_patients(self, fold: int = 1, debug_mode: bool = False):
        patients = Dict()

        dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH)

        if len(dataset_split) - 1 != g.DATASET_FOLDS:
            dataset_split = self._split_dataset()

        patients["test.inter"] = List(dataset_split["test.inter"])
        patients["valid"] = List(dataset_split["fold.{}".format(fold)])

        patients["train"] = List()
        for i in dataset_split:
            if i != "test.inter" and i != "fold.{}".format(fold):
                patients["train"] += List(dataset_split[i])

        if debug_mode:
            patients["train"] = patients["train"][:1]
            # 2 patients in valid and test sets, to debug median score calculation
            patients["valid"] = patients["valid"][:1]
            patients["test.inter"] = patients["test.inter"][:1]

        return patients

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_new_cnn(self, hyper: Dict, in_chan: int, out_chan: int):
        # cnn architecture
        if hyper["cnn"] == "unet.pp.slim":
            cnn = UNetPPSlim
        elif hyper["cnn"] == "unet.slim":
            cnn = UNetSlim
        else:
            Debug.terminate("wrong hyper[cnn] value")
        hyper["cnn"] = cnn(
            in_chan=in_chan,
            out_chan=out_chan,
            slice_thick=hyper["slice.thick"],
            dropout=hyper["dropout"],
        )
        # set multi-GPU
        if GPU.used_count() > 1:
            hyper["cnn"] = DataParallel(hyper["cnn"])
        # to gpu (if gpu available)
        hyper["cnn"] = hyper["cnn"].to(g.DEVICE)

    def _load_exist_cnn(self, cnn_path: str):
        cnn = torch.load(cnn_path)
        if GPU.used_count() > 1:
            cnn = DataParallel(cnn)
        cnn = cnn.to(g.DEVICE)
        return cnn

    def _load_segment_metrics(self, slice_thick: str) -> Dict:
        segment_metrics = Dict()
        for metric in g.METRICS:
            segment_metrics[metric] = SegmentationMetrics(
                metric=metric, slice_thick=slice_thick
            )
            # if GPU.used_count() > 1:
            #     segment_metrics[metric] = DataParallel(segment_metrics[metric])
            segment_metrics[metric] = segment_metrics[metric].to(g.DEVICE)
        return segment_metrics

    def _load_slice_thick(self, hyper: Dict):
        # slice thickness
        if hyper["slice.thick"] != "1mm" and hyper["slice.thick"] != "3mm":
            Debug.terminate("Invalid slice thickness")

    def _load_hyper(self, hyper: Dict) -> None:
        # device name
        if GPU.used_count() < 1:
            hyper["device"] = "cpu"
        else:
            hyper["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]

        # dropout
        hyper["dropout"] = ValueUtils.limit_range(hyper["dropout"], (0.0, 1.0))

        # batch size
        hyper["batch.size"] = ValueUtils.limit_range(hyper["batch.size"], (1, None))
        if GPU.used_count() > 1:
            hyper["batch.size.actual"] = hyper["batch.size"] * GPU.used_count()
        else:
            hyper["batch.size.actual"] = hyper["batch.size"]

        # = 1 will cause error
        hyper["lr.decay.factor"] = ValueUtils.limit_range(
            hyper["lr.decay.factor"], (g.EPS, 1 - g.EPS)
        )

        # augment methods
        hyper["augment.methods"] = List(hyper["augment.methods"])

        # augment lower/upper limit
        hyper["augment.max"] = ValueUtils.limit_range(
            hyper["augment.max"], (1, len(hyper["augment.methods"]))
        )
        hyper["augment.min"] = ValueUtils.limit_range(
            hyper["augment.min"], (1, hyper["augment.max"])
        )

        # loss function parameters
        hyper["loss.weight"] = ValueUtils.limit_range(hyper["loss.weight"], (0.0, 1.0))
        hyper["loss.delta"] = ValueUtils.limit_range(hyper["loss.delta"], (0.0, 1.0))

    def _load_optim_and_scheduler(self, hyper):
        # optimizer (no need to move to cuda)
        if isinstance(hyper["lr.actual"], list):
            lr = hyper["lr.actual"][0]
        else:
            lr = hyper["lr.actual"]
        hyper["optim"] = optim.Adam(params=hyper["cnn"].parameters(), lr=lr)

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

    def _simplify_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()

        for key_name in hyper:
            # "augment.methods" is a list
            if key_name == "augment.methods":
                simple_hyper[key_name] = hyper[key_name].to_str()

            # only save loss function name
            elif key_name == "loss.func":
                simple_hyper[key_name] = "unified.focal.loss"

            # only save cnn name
            elif key_name == "cnn":
                if isinstance(hyper[key_name], DataParallel):
                    cnn = hyper[key_name].module
                else:
                    cnn = hyper[key_name]

                if isinstance(cnn, UNetPPSlim):
                    simple_hyper[key_name] = "unet.pp.slim"
                else:
                    simple_hyper[key_name] = "unet.slim"

            # only save optimizer name
            elif key_name == "optim":
                simple_hyper[key_name] = "adam"

            # only save scheduler name
            elif key_name == "scheduler":
                simple_hyper[key_name] = "reduce.lr.on.plateau"

            # others
            else:
                simple_hyper[key_name] = hyper[key_name]

        simple_hyper = Dict(OrderedDict(sorted(simple_hyper.items())))
        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        for key, value in hyper.items():
            print(key + ":", value)

    # split dataset and save result into json file
    def _split_dataset(self) -> Dict:
        dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH)
        train_patients = List()

        for fold in dataset_split.keys():
            if fold != "test.inter":
                train_patients += List(dataset_split[fold])
                dataset_split.pop(fold)

        random.shuffle(train_patients)

        fold_len = round(len(train_patients) / g.DATASET_FOLDS)
        for fold in range(1, g.DATASET_FOLDS + 1):
            if fold == g.DATASET_FOLDS:
                dataset_split["fold.{}".format(fold)] = train_patients.to_str()
            else:
                dataset_split["fold.{}".format(fold)] = train_patients[
                    :fold_len
                ].to_str()
                train_patients = train_patients[fold_len:]

        Json.save(dataset_split, g.DATASET_SPLIT_JSON_PATH)
        return dataset_split

    def _save_cnn(self, hyper: Dict, save_path: str):
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        if GPU.used_count() > 1:
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
            train_id += "_"
            train_id += g.DELETE_FLAG

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

    # evenly distribute the data into each batch
    def _optimize_batch_size(self, dataset, hyper: Dict):
        if dataset.__len__() > hyper["batch.size"]:
            hyper["batch.size"] = math.ceil(
                dataset.__len__() / (math.ceil(dataset.__len__() / hyper["batch.size"]))
            )
        else:
            hyper["batch.size"] = dataset.__len__()

    # protected function
    def _get_cur_time_str(self) -> str:
        start_time = str(datetime.now().replace(microsecond=0))
        start_time = start_time.replace(":", ".")
        start_time = start_time.replace("-", ".")
        start_time = start_time.replace(" ", ".")
        return start_time

    def _load_hyper_sets_from_json(self, path: str) -> List:
        group_hyper = List()
        origin_hyper_dict = Json.load(path)
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

    # find train result directory full path using train_id
    def _find_train_dir(self, train_id: str) -> str:
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id, "baseline")
        # train id is a baseline
        if os.path.exists(baseline_dir):
            return baseline_dir
        # train id is a iDL
        else:
            for baseline_dir in Explorer.get_sub_folders(
                g.TRAIN_RESULTS_DIR, full_path=True
            ):
                for train_dir in Explorer.get_sub_folders(baseline_dir, full_path=True):
                    if Path(train_dir).name == train_id:
                        return train_dir
            # cant find train_dir
            return None
