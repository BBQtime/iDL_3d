import global_elems as g
import os
import torch
import math
import random
import numpy as np
import torch.nn as nn
from loss_func import UnifiedFocalLoss

# from loss_func import DiceLoss
# from criterion import HybridFocalLoss
from segment_metrics import SegmentationMetrics
from tqdm import tqdm
from itertools import product
from collections import OrderedDict
from torch import optim
from idl_dataset import IDLDataSet
from typing import Union

# from unet_pp import UNetPP
from unet_pp_slim import UNetPPSlim
from datetime import datetime
from nested_dict import NestedDict
from baseline_dataset import BaselineDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SharedTraining:
    def __init__(self):
        self._time_used = None
        self._augment_pct = None
        self._lr = None
        self._lr_actual = None
        self._lr_decay_patience = None
        self._lr_min = None

        self._seg_metrics = NestedDict()
        for metric_type in g.METRICS_LIST:
            self._seg_metrics[metric_type] = SegmentationMetrics(metric_type).to(
                g.DEVICE
            )

    # new hyper are loaded from group of new json files
    # baseline hyper are loaded from exist json file together with exist cnn
    # baseline hyper (cnn/dataset_pct/dataset_seed) only used for iDL
    def _load_hyper(self, hyper: dict, exist_cnn_path: str = None):
        # DROPOUT
        self._dropout = float(hyper["dropout"])
        self._dropout = g.check_limit(self._dropout, 0, 0.9)

        # batch size
        self._batch_size = int(hyper["batch.size"])
        self._batch_size = g.check_limit(self._batch_size, 1, None)

        # actual batch size
        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._batch_size_actual = self._batch_size * used_gpu_count
        else:
            self._batch_size_actual = self._batch_size

        # lr decay factor
        self._lr_decay_factor = float(hyper["lr.decay.factor"])
        # lr_decay_factor=1 will cause error
        self._lr_decay_factor = g.check_limit(self._lr_decay_factor, 0.01, 0.9999999999)

        # augment methods
        self._augment_methods = str(hyper["augment.methods"]).lower()
        if self._augment_methods == "":
            self._augment_methods = []
        else:
            self._augment_methods = g.str_to_list(self._augment_methods)

        # augment lower/upper limit
        self._augment_low_limit = int(hyper["augment.low.limit"])
        self._augment_low_limit = g.check_limit(self._augment_low_limit, 1, 4)

        self._augment_up_limit = int(hyper["augment.up.limit"])
        self._augment_up_limit = g.check_limit(
            self._augment_up_limit, self._augment_low_limit, 4
        )

        # loss function parameters
        weight = float(hyper["loss.weight"])
        weight = g.check_limit(weight, 0, 1)
        delta = float(hyper["loss.delta"])
        delta = g.check_limit(delta, 0, 1)
        gamma = float(hyper["loss.gamma"])
        asym = bool(hyper["loss.asym"])

        # self._loss_func = DiceLoss().to(g.DEVICE)
        # self._loss_func = HybridFocalLoss().to(g.DEVICE)
        self._loss_func = UnifiedFocalLoss(
            weight=weight,
            delta=delta,
            gamma=gamma,
            asym=asym,
        ).to(g.DEVICE)

        # load cnn
        self._cnn = self._load_cnn(exist_cnn_path)

        # optimizer (no need to move to cuda)
        self._optim = optim.Adam(params=self._cnn.parameters(), lr=self._lr_actual)

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        self._scheduler = ReduceLROnPlateau(
            optimizer=self._optim,
            mode="min",
            factor=self._lr_decay_factor,  # "factor=1" will cause an error
            patience=self._lr_decay_patience,
            min_lr=self._lr_min,
        )

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_cnn(self, exist_cnn_path: str = None):
        # new model
        if exist_cnn_path is None:
            # cnn = UNetPP(dropout=self._dropout).to(g.DEVICE)
            cnn = UNetPPSlim(in_chan=4, out_chan=3, dropout=self._dropout).to(g.DEVICE)

        # exist cnn
        else:
            # load state dict only
            if g.CNN_STATE_DICT_ONLY:
                # cnn = UNetPP(dropout=self._dropout).to(g.DEVICE)
                cnn = UNetPPSlim(in_chan=4, out_chan=3, dropout=self._dropout).to(
                    g.DEVICE
                )
                cnn.load_state_dict(torch.load(exist_cnn_path))

            # load entire cnn
            else:
                cnn = torch.load(exist_cnn_path).to(g.DEVICE)

        # set multi-GPU
        if g.used_gpu_count() > 1:
            cnn = nn.DataParallel(cnn).to(g.DEVICE)
        return cnn

    def _print_hyper(self, print_dict: NestedDict):
        if torch.cuda.device_count() < 1:
            print_dict["device"] = "cpu"
        else:
            print_dict["device"] = "gpu: " + os.environ["CUDA_VISIBLE_DEVICES"]
        print_dict["lr"] = self._lr
        print_dict["lr.actual"] = self._lr_actual
        print_dict["lr.decay.factor"] = self._lr_decay_factor
        print_dict["lr.decay.patience"] = self._lr_decay_patience
        print_dict["lr.min"] = self._lr_min
        print_dict["batch.size"] = self._batch_size
        print_dict["batch.size.actual"] = self._batch_size_actual
        print_dict["augment.pct"] = self._augment_pct
        print_dict["augment.methods"] = self._augment_methods
        print_dict["augment.low.limit"] = self._augment_low_limit
        print_dict["augment.up.limit"] = self._augment_up_limit
        print_dict["loss.weight"] = self._loss_func.weight
        print_dict["loss.delta"] = self._loss_func.delta
        print_dict["loss.gamma"] = self._loss_func.gamma
        print_dict["loss.asym"] = self._loss_func.asym

        print_dict = OrderedDict(sorted(print_dict.items()))
        for key, value in print_dict.items():
            print(key + ":", value)

    def _save_hyper(self, json_path: str, hyper_dict: NestedDict):
        if torch.cuda.device_count() < 1:
            hyper_dict["device"] = "cpu"
        else:
            hyper_dict["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]
        hyper_dict["time.used"] = self._time_used
        hyper_dict["lr.actual"] = self._lr_actual
        hyper_dict["lr.decay.factor"] = self._lr_decay_factor
        hyper_dict["lr.decay.patience"] = self._lr_decay_patience
        hyper_dict["lr.min"] = self._lr_min
        hyper_dict["batch.size"] = self._batch_size
        hyper_dict["batch.size.actual"] = self._batch_size_actual
        hyper_dict["augment.pct"] = self._augment_pct
        hyper_dict["augment.methods"] = g.list_to_str(self._augment_methods)
        hyper_dict["augment.low.limit"] = self._augment_low_limit
        hyper_dict["augment.up.limit"] = self._augment_up_limit
        hyper_dict["loss.weight"] = self._loss_func.weight
        hyper_dict["loss.delta"] = self._loss_func.delta
        hyper_dict["loss.gamma"] = self._loss_func.gamma
        hyper_dict["loss.asym"] = self._loss_func.asym
        hyper_dict["loss.func"] = "unified.focal.loss"
        hyper_dict["optim"] = "adam"
        hyper_dict["scheduler"] = "reduce.lr.on.plateau"
        # save dict
        g.save_json(data=hyper_dict, path=json_path)

    def _load_dataset(self, fold: int, debug_mode: bool = False):
        json_data = g.load_json(g.DATASET_SPLITTING_JSON)

        if len(json_data) - 1 != g.DATASET_K_FOLDS:
            json_data = self.__split_dataset()

        test_patients = g.str_to_list(json_data["test.patients"])
        valid_patients = g.str_to_list(json_data["fold.{}".format(fold)])

        train_patients = []
        for i in json_data:
            if i != "test.patients" and i != "fold.{}".format(fold):
                train_patients += g.str_to_list(json_data[i])

        # if 1:
        #     print(set(train_patients) & set(valid_patients))

        if debug_mode:
            train_patients = train_patients[: self._batch_size_actual]
            valid_patients = valid_patients[: self._batch_size_actual]
            test_patients = test_patients[: self._batch_size_actual]

        return train_patients, valid_patients, test_patients

    # split dataset and save result into json file
    def __split_dataset(self):
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

    def _save_cnn(self, save_path: str):
        if not save_path.endswith(".pt"):
            save_path += ".pt"

        # save state dict only
        if g.CNN_STATE_DICT_ONLY:
            if g.used_gpu_count() > 1:
                torch.save(self._cnn.module.state_dict(), save_path)
            else:
                torch.save(self._cnn.state_dict(), save_path)

        # save entire cnn
        else:
            if g.used_gpu_count() > 1:
                torch.save(self._cnn.module, save_path)
            else:
                torch.save(self._cnn, save_path)
        return

    # train_id = start_time + train_remark
    def _init_train_id(
        self,
        train_remark: str,
        debug_mode: bool,
        hyper_json_path: str,
        hyper: dict,
    ):

        train_id = self._get_cur_time_str()

        if debug_mode:
            train_id += "_debug.mode.delete.this"

        if train_remark != "":
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
                train_id += "=" + str(hyper[key]).replace("_", ".")

        return train_id

    def _optimize_batch_size(self, dataset: Union[BaselineDataSet, IDLDataSet]):
        dataset_len = dataset.__len__()
        if dataset_len > self._batch_size_actual:
            self._batch_size_actual = math.ceil(
                dataset_len / (math.ceil(dataset_len / self._batch_size_actual))
            )
        else:
            self._batch_size_actual = dataset_len  # self._batch_size_actual

    def _inference_single_patient(self, patient: str, unetpp_output: int = 3) -> dict:
        result = NestedDict()
        dataset = BaselineDataSet(patient_list=[patient])

        origin_gtvs = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient))
        )

        self._cnn.eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            inputs, labels = dataset.get_item(patient=patient)
            inputs = torch.unsqueeze(inputs.to(g.DEVICE), dim=0)
            labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
            outputs = self._cnn.forward(inputs)[unetpp_output]
            # squeeze batch
            outputs = torch.squeeze(outputs, dim=0).cpu().numpy()

            # get gtvt/gtvn/gtvs
            result["gtvt"] = outputs[1]
            result["gtvn"] = outputs[2]
            result["gtvs"] = result["gtvt"] + result["gtvn"]
            result["gtvs"] = np.where(result["gtvs"] > 1, 1, result["gtvs"])

            # pad and crop to original size
            for i in ["gtvt", "gtvn", "gtvs"]:
                result[i] = g.central_pad(result[i], origin_gtvs.shape)
                result[i] = g.central_crop(result[i], origin_gtvs.shape)

        for metric_type in g.METRICS_LIST:
            result[metric_type] = self._seg_metrics[metric_type](
                result["gtvs"], origin_gtvs
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
                cur_hyper[hyper_keys[i]] = cur_values[i]
            group_hyper.append(cur_hyper)

        return group_hyper
