from custom import Global as g
import os
import torch
import math
import random
import numpy as np
from torch.nn import DataParallel
from numpy import ndarray
from segment_metrics import SegmentationMetrics
from itertools import product
from collections import OrderedDict
from torch import optim
from unet_pp_slim import UNetPPSlim
from unet_slim import UNetSlim
from datetime import datetime
from baseline_dataset import BaselineDataSet
from idl_gtvn_dataset import IDLGTVnDataSet
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
            if isinstance(hyper["cnn"], DataParallel):
                hyper["cnn"] = hyper["cnn"].module

            if hyper["cnn"] == "unet.pp.slim" or isinstance(hyper["cnn"], UNetPPSlim):
                if hyper["train.type"] == "baseline":
                    in_chan = 4
                    out_chan = 3
                else:
                    in_chan = 5
                    out_chan = 2
                hyper["cnn"] = UNetPPSlim(
                    in_chan=in_chan, out_chan=out_chan, dropout=hyper["dropout"]
                ).to(g.DEVICE)

            elif hyper["cnn"] == "unet.slim" or isinstance(hyper["cnn"], UNetSlim):
                hyper["cnn"] = UNetSlim(
                    in_chan=5, out_chan=2, dropout=hyper["dropout"]
                ).to(g.DEVICE)

        # existing model
        else:
            hyper["cnn"] = torch.load(cnn_path).to(g.DEVICE)

        # set multi-GPU
        if GPU.used_count() > 1:
            hyper["cnn"] = DataParallel(hyper["cnn"]).to(g.DEVICE)

    def __simplify_hyper(self, hyper: Dict) -> Dict:
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
                if isinstance(hyper[key_name], UNetPPSlim):
                    simple_hyper[key_name] = "unet.pp.slim"
                elif isinstance(hyper[key_name], UNetSlim):
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
        simple_hyper = self.__simplify_hyper(hyper)
        for key, value in simple_hyper.items():
            print(key + ":", value)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self.__simplify_hyper(hyper)
        Json.save(data=simple_hyper, path=json_path)

    # split dataset and save result into json file
    def _split_dataset(self) -> Dict:
        dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH)
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

    def _patient_inference(
        self,
        patient: str,
        hyper: Dict,
        inference_type: str,  # baseline/idl_gtvt/idl_gtvn
        idl_gtvt_masked_label: ndarray = None,  # gtvt post processing
        idl_gtvn_baseline_epoch_dir: str = None,  # idl_gtvn dataset needs this
    ) -> Dict:

        if inference_type != "idl_gtvt" and inference_type != "idl_gtvn":
            inference_type = "baseline"

        # result structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
        result = Dict()
        # original labels
        origin = Dict()

        if inference_type == "baseline" or inference_type == "idl_gtvt":
            dataset = BaselineDataSet(patients=[patient])
        else:
            dataset = IDLGTVnDataSet(
                patients=[patient],
                baseline_epoch_dir=idl_gtvn_baseline_epoch_dir,
                random_click=False,
            )

        # load gtvs
        if inference_type == "baseline":
            origin["gtvs"] = Nii.load(
                os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVs.nii".format(patient)),
                binary=True,
            )

        # load gtvt
        if inference_type == "baseline" or inference_type == "idl_gtvt":
            origin["gtvt"] = Nii.load(
                os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVt.nii".format(patient)),
                binary=True,
            )

        # load gtvn
        if inference_type == "baseline" or inference_type == "idl_gtvn":
            origin["gtvn"] = Nii.load(
                os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVn.nii".format(patient)),
                binary=True,
            )

        # get pred
        hyper["cnn"].eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            input_imgs, labels = dataset.get_item(patient=patient)
            input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)
            labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
            preds = hyper["cnn"].forward(input_imgs)
            # squeeze "batch" channel
            preds = torch.squeeze(preds, dim=0).cpu().numpy()

        if inference_type == "baseline":
            result["gtvt"]["pred"] = preds[1]
            result["gtvn"]["pred"] = preds[2]
            result["gtvs"]["pred"] = np.maximum(preds[1], preds[2])
            gtv_list = ["gtvs", "gtvt", "gtvn"]

        elif inference_type == "idl_gtvt":
            result["gtvt"]["pred"] = preds[1]
            gtv_list = ["gtvt"]

        elif inference_type == "idl_gtvn":
            result["gtvn"]["pred"] = preds[1]
            input_imgs = torch.squeeze(input_imgs, dim=0).cpu().numpy()
            # imput_imgs: pred, clicks, ...
            result["gtvn"]["clicks"] = input_imgs[1]
            gtv_list = ["gtvn"]

        # pad and crop to original size
        # pred
        for gtv in gtv_list:
            result[gtv]["pred"] = Img.central_pad(
                result[gtv]["pred"], origin[gtv].shape
            )
            result[gtv]["pred"] = Img.central_crop(
                result[gtv]["pred"], origin[gtv].shape
            )
        # clicks (if idl_gtvn)
        if inference_type == "idl_gtvn":
            result["gtvn"]["clicks"] = Img.central_pad(
                result["gtvn"]["clicks"], origin[gtv].shape
            )
            result["gtvn"]["clicks"] = Img.central_crop(
                result["gtvn"]["clicks"], origin[gtv].shape
            )

        # idl post processing (before calculate scores)
        if inference_type == "idl_gtvt" and idl_gtvt_masked_label is not None:
            cc_list = Img.connected_components(result["gtvt"]["pred"])
            result["gtvt"]["pred"] = np.zeros_like(result["gtvt"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * idl_gtvt_masked_label).sum() > 0:
                    result["gtvt"]["pred"] = np.maximum(result["gtvt"]["pred"], cur_cc)

        # calculate inference scores
        for gtv in gtv_list:
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

    def _find_result_dir(self, result_id: str) -> str:
        # find result directory full path using result id
        result_dir = None
        for i in Explorer.walk_sub_dirs(g.TRAIN_RESULTS_DIR, key_word=result_id):
            # remove "/" if str endswith it
            if i.endswith("/"):
                i = i[:-1]
            if i.endswith(result_id):
                result_dir = i
                break
        return result_dir
