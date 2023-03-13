import global_elems as g
import os
import torch
import math
import random
import numpy as np
import torch.nn as nn
from segment_metrics import SegmentationMetrics
from itertools import product
from collections import OrderedDict
from torch import optim
from idl_dataset import IDLDataSet
from typing import Union
from unet_pp_slim import UNetPPSlim
from datetime import datetime
from nested_dict import NestedDict
from baseline_dataset import BaselineDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SharedTraining:

    # new hyper are loaded from group of new json files
    # baseline hyper are loaded from exist json file together with exist cnn
    # baseline hyper (cnn/dataset_pct/dataset_seed) only used for iDL
    def _load_hyper(self, hyper: dict, exist_cnn_path: str = ""):
        # segmentation metrics
        hyper["metrics"] = NestedDict()
        for metric_type in g.METRICS_LIST:
            hyper["metrics"][metric_type] = SegmentationMetrics(metric_type).to(
                g.DEVICE
            )

        # device name
        if torch.cuda.device_count() < 1:
            hyper["device"] = "cpu"
        else:
            hyper["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]

        # dropout
        hyper["dropout"] = float(hyper["dropout"])
        hyper["dropout"] = g.check_limit(hyper["dropout"], 0, 0.9)

        # batch size
        # new training, hyper["batch.size"] is a number
        # inference, hyper["batch.size"] is a dict
        if not isinstance(hyper["batch.size"], dict):
            hyper["batch.size"] = {"init": int(hyper["batch.size"])}
        hyper["batch.size"]["init"] = g.check_limit(
            hyper["batch.size"]["init"], 1, None
        )

        # actual batch size
        if g.used_gpu_count() > 1:
            hyper["batch.size"]["actual"] = int(
                hyper["batch.size"]["init"] * g.used_gpu_count()
            )
        else:
            hyper["batch.size"]["actual"] = int(hyper["batch.size"]["init"])

        # lr decay factor
        hyper["lr"]["decay.factor"] = float(hyper["lr"]["decay.factor"])
        # lr_decay_factor=1 will cause error
        hyper["lr"]["decay.factor"] = g.check_limit(
            hyper["lr"]["decay.factor"], 0.01, 0.9999999999
        )

        # augment methods
        hyper["augment"]["methods"] = str(hyper["augment"]["methods"]).lower()
        if hyper["augment"]["methods"] == "":
            hyper["augment"]["methods"] = []
        else:
            hyper["augment"]["methods"] = g.str_to_list(hyper["augment"]["methods"])

        # augment lower/upper limit
        hyper["augment"]["low.limit"] = int(hyper["augment"]["low.limit"])
        hyper["augment"]["low.limit"] = g.check_limit(
            hyper["augment"]["low.limit"], 1, 4
        )

        hyper["augment"]["up.limit"] = int(hyper["augment"]["up.limit"])
        hyper["augment"]["up.limit"] = g.check_limit(
            hyper["augment"]["up.limit"], hyper["augment"]["low.limit"], 4
        )

        # loss function parameters
        hyper["loss"]["weight"] = float(hyper["loss"]["weight"])
        hyper["loss"]["weight"] = g.check_limit(hyper["loss"]["weight"], 0, 1)

        hyper["loss"]["delta"] = float(hyper["loss"]["delta"])
        hyper["loss"]["delta"] = g.check_limit(hyper["loss"]["delta"], 0, 1)

        hyper["loss"]["gamma"] = float(hyper["loss"]["gamma"])

        hyper["loss"]["asym"] = bool(hyper["loss"]["asym"])

        # load cnn
        self._load_cnn(hyper, exist_cnn_path)

        # optimizer (no need to move to cuda)
        if isinstance(hyper["lr"]["actual"], list):
            actual_lr = hyper["lr"]["actual"][0]
        else:
            actual_lr = hyper["lr"]["actual"]
        hyper["optim"] = optim.Adam(params=hyper["cnn"].parameters(), lr=actual_lr)

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
    def _load_cnn(self, hyper: NestedDict, exist_cnn_path: str = ""):
        # new model
        if exist_cnn_path == "":
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
                hyper["cnn"].load_state_dict(torch.load(exist_cnn_path))

            # load entire cnn
            else:
                hyper["cnn"] = torch.load(exist_cnn_path).to(g.DEVICE)

        # set multi-GPU
        if g.used_gpu_count() > 1:
            hyper["cnn"] = nn.DataParallel(hyper["cnn"]).to(g.DEVICE)

    def __get_simple_hyper(self, hyper: NestedDict) -> NestedDict:
        simple_hyper = NestedDict()
        for i in hyper:
            if i == "metrics":
                pass

            elif i == "augment":
                simple_hyper[i] = hyper[i].copy()
                simple_hyper[i]["methods"] = g.list_to_str(simple_hyper[i]["methods"])

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

    def _print_hyper(self, hyper: NestedDict):
        simple_hyper = self.__get_simple_hyper(hyper)
        for key, value in simple_hyper.items():
            print(key + ":", value)

    def _save_hyper(self, hyper: NestedDict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        g.save_json(data=simple_hyper, path=json_path)

    # split dataset and save result into json file
    def _split_dataset(self) -> dict:
        dataset_split = g.load_json(g.DATASET_SPLITTING_JSON)
        train_patients = []

        for fold in g.get_dict_keys(dataset_split):
            if fold != "test.patients":
                train_patients += g.str_to_list(dataset_split[fold])
                dataset_split.pop(fold)

        random.shuffle(train_patients)

        fold_len = round(len(train_patients) / g.DATASET_K_FOLDS)
        for fold in range(1, g.DATASET_K_FOLDS + 1):
            if fold == g.DATASET_K_FOLDS:
                dataset_split["fold.{}".format(fold)] = g.list_to_str(train_patients)
            else:
                dataset_split["fold.{}".format(fold)] = g.list_to_str(
                    train_patients[:fold_len]
                )
                train_patients = train_patients[fold_len:]

        g.save_json(dataset_split, g.DATASET_SPLITTING_JSON)
        return dataset_split

    def _save_cnn(self, hyper: NestedDict, save_path: str):
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
        hyper: NestedDict,
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
        origin_hyper_dict = g.load_json(hyper_json_path)
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
        self, hyper: NestedDict, dataset: Union[BaselineDataSet, IDLDataSet]
    ):
        dataset_len = dataset.__len__()
        if dataset_len > hyper["batch.size"]["actual"]:
            hyper["batch.size"]["actual"] = math.ceil(
                dataset_len / (math.ceil(dataset_len / hyper["batch.size"]["actual"]))
            )
        else:
            hyper["batch.size"]["actual"] = dataset_len

    def _inference_single_patient(
        self, patient: str, hyper: NestedDict, unetpp_output: int = 3
    ) -> NestedDict:
        # result structure: gtvs/gtvt/gvtn → pred/dsc/msd/hd95
        result = NestedDict()

        dataset = BaselineDataSet(patient_list=[patient])

        origin = NestedDict()
        # load gtvt gtvs
        for i in ["s", "t"]:
            origin["gtv" + i] = g.load_nii(
                os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTV{}.nii".format(patient, i)),
                binary=True,
            )
        # load gtvn
        gtvn_path = os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(patient))
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
            result["gtvn"]["pred"] = outputs[2]
            result["gtvs"]["pred"] = np.maximum(outputs[1], outputs[2])

            # pad and crop to original size
            for i in ["gtvs", "gtvt", "gtvn"]:
                result[i]["pred"] = g.central_pad(result[i]["pred"], origin[i].shape)
                result[i]["pred"] = g.central_crop(result[i]["pred"], origin[i].shape)

        # calculate segment scores
        for i in ["gtvs", "gtvt", "gtvn"]:
            for metric_type in g.METRICS_LIST:
                result[i][metric_type] = hyper["metrics"][metric_type](
                    result[i]["pred"], origin[i]
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
        group_hyper = []
        origin_hyper_dict = g.load_json(json_path)
        hyper_keys = g.get_dict_keys(origin_hyper_dict)

        # make sure all values of hyper dict are "list" type
        for i in origin_hyper_dict:
            if not isinstance(origin_hyper_dict[i], list):
                origin_hyper_dict[i] = [origin_hyper_dict[i]]

        # get all cartesian products of hyper dict values
        for cur_values in product(*origin_hyper_dict.values()):
            # create current hyper param combination
            cur_hyper = NestedDict()
            for i in range(len(cur_values)):
                # copy the value if it is a dict, avoid using same memory
                if isinstance(cur_values[i], dict):
                    cur_hyper[hyper_keys[i]] = cur_values[i].copy()
                else:
                    cur_hyper[hyper_keys[i]] = cur_values[i]
            group_hyper.append(cur_hyper)

        return group_hyper
