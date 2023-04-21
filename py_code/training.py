from custom import Global as g
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
from unet_pp_slim import UNetPPSlim
from unet_slim import UNetSlim
from datetime import datetime
from baseline_dataset import BaselineDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom import Json
from custom import Dict
from custom import List
from custom import GPU
from custom import Img
from custom import Nii
from custom import Value
from custom import Explorer


class Training:
    def __init__(self):
        # segmentation metrics
        self._metrics = Dict()
        for metric in g.METRICS:
            self._metrics[metric] = SegmentationMetrics(metric).to(g.DEVICE)

    # new hyper are loaded from group of new json files
    # baseline hyper are loaded from exist json file together with exist cnn
    # baseline hyper (cnn/dataset_pct/dataset_seed) only used for iDL
    def _load_hyper(self, hyper: Dict, cnn_path: str = None) -> None:

        # device name
        if torch.cuda.device_count() < 1:
            hyper["device"] = "cpu"
        else:
            hyper["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]

        # dropout
        hyper["dropout"] = Value.limit_range(hyper["dropout"], (0.0, 1.0))

        # batch size
        hyper["batch.size"] = Value.limit_range(hyper["batch.size"], (1, None))
        hyper["batch.size.actual"] = hyper["batch.size"] * GPU.used_count()

        # = 1 will cause error
        hyper["lr.decay.factor"] = Value.limit_range(
            hyper["lr.decay.factor"], (g.EPS, 1 - g.EPS)
        )

        # augment methods
        hyper["augment.methods"] = List(hyper["augment.methods"])

        # augment lower/upper limit
        hyper["augment.max"] = Value.limit_range(
            hyper["augment.max"], (1, len(hyper["augment.methods"]))
        )
        hyper["augment.min"] = Value.limit_range(
            hyper["augment.min"], (1, hyper["augment.max"])
        )

        # loss function parameters
        hyper["loss.weight"] = Value.limit_range(hyper["loss.weight"], (0.0, 1.0))
        hyper["loss.delta"] = Value.limit_range(hyper["loss.delta"], (0.0, 1.0))

        # load cnn
        self._load_cnn(hyper=hyper, cnn_path=cnn_path)

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

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_cnn(self, hyper: Dict, cnn_path: str = None):
        # new model
        if cnn_path == "" or cnn_path is None:
            if hyper["cnn"] == "unet.pp.slim" or isinstance(hyper["cnn"], UNetPPSlim):
                hyper["cnn"] = UNetPPSlim(
                    in_chan=4, out_chan=3, dropout=hyper["dropout"]
                ).to(g.DEVICE)
            elif hyper["cnn"] == "unet.slim" or isinstance(hyper["cnn"], UNetSlim):
                hyper["cnn"] = UNetSlim(
                    in_chan=6, out_chan=2, dropout=hyper["dropout"]
                ).to(g.DEVICE)

        # exist cnn
        else:
            hyper["cnn"] = torch.load(cnn_path).to(g.DEVICE)
        # set multi-GPU
        if GPU.used_count() > 1:
            hyper["cnn"] = nn.DataParallel(hyper["cnn"]).to(g.DEVICE)

    def __get_simple_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for cur_key in hyper:
            if cur_key == "augment.methods":
                simple_hyper[cur_key] = hyper[cur_key].to_str()

            elif cur_key == "loss.func":
                simple_hyper[cur_key] = "unified.focal.loss"

            elif cur_key == "cnn":
                if isinstance(hyper[cur_key], UNetPPSlim):
                    simple_hyper[cur_key] = "unet.pp.slim"
                elif isinstance(hyper[cur_key], UNetSlim):
                    simple_hyper[cur_key] = "unet.slim"

            elif cur_key == "optim":
                simple_hyper[cur_key] = "adam"

            elif cur_key == "scheduler":
                simple_hyper[cur_key] = "reduce.lr.on.plateau"

            else:
                simple_hyper[cur_key] = hyper[cur_key]

        simple_hyper = Dict(OrderedDict(sorted(simple_hyper.items())))
        # for i in simple_hyper:
        #     if isinstance(simple_hyper[i], dict):
        #         simple_hyper[i] = Dict(OrderedDict(sorted(simple_hyper[i].items())))

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

    # evenly distribute the data into each batch
    def _optimize_batch_size(self, dataset, hyper: Dict):
        if dataset.__len__() > hyper["batch.size"]:
            hyper["batch.size"] = math.ceil(
                dataset.__len__() / (math.ceil(dataset.__len__() / hyper["batch.size"]))
            )
        else:
            hyper["batch.size"] = dataset.__len__()

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
        origin["gtvt"] = Nii.load(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )

        if not gtvt_only:
            # load gtvs
            origin["gtvs"] = Nii.load(
                os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)),
                binary=True,
            )
            # load gtvn
            gtvn_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(patient)
            )
            if os.path.exists(gtvn_path):
                origin["gtvn"] = Nii.load(gtvn_path, binary=True)
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
                result[gtv]["pred"] = Img.central_pad(
                    result[gtv]["pred"], origin[gtv].shape
                )
                result[gtv]["pred"] = Img.central_crop(
                    result[gtv]["pred"], origin[gtv].shape
                )

        # idl post processing
        if masked_label is not None and gtvt_only:
            cc_list = Img.connected_components(result["gtvt"]["pred"])
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

    def _find_result_folder(self, result_id: str) -> str:
        # find result folder full path using result id
        result_folder = None
        for i in Explorer.walk_sub_folders(g.TRAIN_RESULTS_FOLDER, key_word=result_id):
            # remove "/" if str endswith it
            if i.endswith("/"):
                i = i[:-1]
            if i.endswith(result_id):
                result_folder = i
                break
        return result_folder
