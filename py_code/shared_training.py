import global_elems as g
import os
import torch
import math
import criterion as crit
import torch.nn as nn
from collections import OrderedDict
from torch import optim
from idl_dataset import IDLDataSet
from typing import Tuple, Union
from unet_pp import UNetPP
from tqdm import tqdm
from datetime import datetime
from nested_dict import NestedDict
from torch.utils.data import DataLoader
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

        self._score_funcs = NestedDict()
        for score_type in ["dsc", "msd", "hd95"]:
            self._score_funcs[score_type] = crit.ScoreFunction(
                score_type=score_type, dim="3d"
            ).to(g.DEVICE)

    # can be shared with baseline_visualize
    def _load_batch_size(self, hyper_dict: dict) -> tuple[int, int]:
        batch_size = int(hyper_dict["batch.size"])
        batch_size = g.check_limit(batch_size, 1, None)

        # used_gpu_count = g.used_gpu_count()
        # if used_gpu_count > 1:
        #     batch_size_actual = batch_size * used_gpu_count
        # else:
        #     batch_size_actual = batch_size

        batch_size_actual = batch_size

        self._batch_size_actual = g.check_limit(
            batch_size_actual, None, g.MAX_BATCH_SIZE
        )

        return batch_size, batch_size_actual

    # new hyper are loaded from group of new json files
    # baseline hyper are loaded from exist json file together with exist cnn
    # baseline hyper (cnn/dataset_pct/dataset_seed) only used for iDL
    def _load_cur_hyper(self, cur_hyper_dict: dict, exist_cnn_path: str = None):
        # DROPOUT
        self._dropout = float(cur_hyper_dict["dropout"])
        self._dropout = g.check_limit(self._dropout, 0.0, 0.9)

        # batch size
        self._batch_size, self._batch_size_actual = self._load_batch_size(
            cur_hyper_dict
        )

        # lr decay factor
        self._lr_decay_factor = float(cur_hyper_dict["lr.decay.factor"])
        # lr_decay_factor=1.0 will cause error
        self._lr_decay_factor = g.check_limit(self._lr_decay_factor, 0.01, 0.9999999999)

        # augment method
        self._augment_method = str(cur_hyper_dict["augment.method"]).lower()
        if (
            self._augment_method != "combine"
            and self._augment_method != "scale"
            and self._augment_method != "translate"
            and self._augment_method != "rotate"
            and self._augment_method != "elastic"
        ):
            self._augment_method = None

        # augment lower/upper limit
        self._augment_low_limit = int(cur_hyper_dict["augment.low.limit"])
        self._augment_low_limit = g.check_limit(self._augment_low_limit, 1, 4)

        self._augment_up_limit = int(cur_hyper_dict["augment.up.limit"])
        self._augment_up_limit = g.check_limit(
            self._augment_up_limit, self._augment_low_limit, 4
        )

        # loss function parameters
        hybrid_weight = float(cur_hyper_dict["loss.hybrid.weight"])
        hybrid_weight = g.check_limit(hybrid_weight, 0.0, 1.0)

        tversky_fp_weight = float(cur_hyper_dict["loss.tversky.fp.weight"])
        tversky_fp_weight = g.check_limit(tversky_fp_weight, 0.0, 1.0)
        tversky_fore_power = float(cur_hyper_dict["loss.tversky.fore.power"])
        tversky_fore_weight = float(cur_hyper_dict["loss.tversky.fore.weight"])

        bce_back_power = float(cur_hyper_dict["loss.bce.back.power"])
        bce_fore_weight = float(cur_hyper_dict["loss.bce.fore.weight"])
        bce_fore_weight = g.check_limit(bce_fore_weight, 0.0, 1.0)

        # self._loss_func = crit.Avg2dDiceLoss().to(g.DEVICE)
        self._loss_func = crit.HybridFocalLoss(
            dim="3d",
            hybrid_weight=hybrid_weight,
            tversky_fore_weight=tversky_fore_weight,
            tversky_fore_power=tversky_fore_power,
            tversky_fp_weight=tversky_fp_weight,
            bce_fore_weight=bce_fore_weight,
            bce_back_power=bce_back_power,
        ).to(g.DEVICE)

        # load cnn
        self._cnn = self._load_cnn(
            cnn_name=str(cur_hyper_dict["cnn.name"]),  # unet or unet++
            exist_cnn_path=exist_cnn_path,
        )

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
            factor=self._lr_decay_factor,  # "factor=1.0" will cause an error
            patience=self._lr_decay_patience,
            min_lr=self._lr_min,
        )

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_cnn(self, cnn_name: str, exist_cnn_path: str = None):
        # new model
        if exist_cnn_path is None:
            if cnn_name == "unet++":
                cnn = UNetPP(dropout=self._dropout).to(g.DEVICE)
            else:
                cnn = UNetPP(dropout=self._dropout).to(g.DEVICE)

        # exist cnn
        else:
            # load state dict only
            if g.CNN_STATE_DICT_ONLY:
                if cnn_name == "unet++":
                    cnn = UNetPP(dropout=self._dropout).to(g.DEVICE)
                else:
                    cnn = UNetPP(dropout=self._dropout).to(g.DEVICE)
                cnn.load_state_dict(torch.load(exist_cnn_path))

            # load entire cnn
            else:
                cnn = torch.load(exist_cnn_path).to(g.DEVICE)

        # set multi-GPU
        if g.used_gpu_count() > 1:
            cnn = nn.DataParallel(cnn).to(g.DEVICE)
        return cnn

    def __get_cnn_name(self):
        if g.used_gpu_count() > 1:
            cnn = self._cnn.module
        else:
            cnn = self._cnn
        if isinstance(cnn, UNetPP):
            return "unet++"
        else:
            return None

    def _print_hyper(self, print_dict: NestedDict):
        print_dict["device:"] = g.DEVICE
        print_dict["cnn name:"] = self.__get_cnn_name()
        print_dict["lr:"] = self._lr
        print_dict["lr actual:"] = self._lr_actual
        print_dict["lr decay factor:"] = self._lr_decay_factor
        print_dict["lr decay patience:"] = self._lr_decay_patience
        print_dict["lr min:"] = self._lr_min
        print_dict["batch size:"] = self._batch_size
        print_dict["batch size actual:"] = self._batch_size_actual
        print_dict["augment percent:"] = self._augment_pct
        print_dict["augment method:"] = self._augment_method
        print_dict["augment lower limit:"] = self._augment_low_limit
        print_dict["augment upper limit:"] = self._augment_up_limit
        print_dict["loss hybrid weight:"] = self._loss_func.hybrid_weight
        print_dict["loss tversky fore weight:"] = self._loss_func.tversky_fore_weight
        print_dict["loss tversky fore power:"] = self._loss_func.tversky_fore_power
        print_dict["loss tversky fp weight:"] = self._loss_func.tversky_fp_weight
        print_dict["loss bce fore weight:"] = self._loss_func.bce_fore_weight
        print_dict["loss bce back power:"] = self._loss_func.bce_back_power

        print_dict = OrderedDict(sorted(print_dict.items()))
        for key, value in print_dict.items():
            print(key, value)

    def _save_hyper(self, json_path: str, hyper_dict: NestedDict):
        hyper_dict["time.used"] = str(self._time_used)
        hyper_dict["device"] = str(g.DEVICE)
        hyper_dict["lr.actual"] = self._lr_actual
        hyper_dict["lr.decay.factor"] = self._lr_decay_factor
        hyper_dict["lr.decay.patience"] = self._lr_decay_patience
        hyper_dict["lr.min"] = self._lr_min
        hyper_dict["batch.size"] = self._batch_size
        hyper_dict["batch.size.actual"] = self._batch_size_actual
        hyper_dict["augment.pct"] = self._augment_pct
        hyper_dict["augment.method"] = self._augment_method
        hyper_dict["augment.low.limit"] = self._augment_low_limit
        hyper_dict["augment.up.limit"] = self._augment_up_limit
        hyper_dict["loss.hybrid.weight"] = self._loss_func.hybrid_weight
        hyper_dict["loss.tversky.fore.weight"] = self._loss_func.tversky_fore_weight
        hyper_dict["loss.tversky.fore.power"] = self._loss_func.tversky_fore_power
        hyper_dict["loss.tversky.fp.weight"] = self._loss_func.tversky_fp_weight
        hyper_dict["loss.bce.fore.weight"] = self._loss_func.bce_fore_weight
        hyper_dict["loss.bce.back.power"] = self._loss_func.bce_back_power
        hyper_dict["loss.func"] = "hybrid.focal.loss"
        hyper_dict["optim"] = "adam"
        hyper_dict["scheduler"] = "reduce.lr.on.plateau"
        hyper_dict["cnn.name"] = self.__get_cnn_name()
        # save dict
        g.save_json(data=hyper_dict, path=json_path)

    def _load_dataset(self, debug_mode: bool = False):
        # if json file doesnt exist, spilt dataset and regenerate json file
        if not os.path.exists(g.DATASET_SPLITTING_JSON):
            self.__split_dataset()

        json_data = g.load_json(g.DATASET_SPLITTING_JSON)
        train_patient_list = json_data["train.patient.list"]
        valid_patient_list = json_data["valid.patient.list"]
        test_patient_list = json_data["test.patient.list"]

        if debug_mode:
            train_patient_list = train_patient_list[:2]
            valid_patient_list = valid_patient_list[:2]
            test_patient_list = test_patient_list[:2]

        return train_patient_list, valid_patient_list, test_patient_list

    # split dataset and save result into json file
    def __split_dataset(self):
        json_data = g.load_json(os.path.join(g.PROJ_PATH, "settings.json"))

        dataset_split_seed = int(json_data["dataset.split.seed"])
        train_pct = float(json_data["dataset.train.pct"])
        train_pct = g.check_limit(train_pct, 0.5, 0.95)
        valid_pct = float(json_data["dataset.valid.pct"])
        valid_pct = g.check_limit(valid_pct, 0.05, 0.2)
        test_pct = float(json_data["dataset.test.pct"])
        test_pct = g.check_limit(test_pct, 0.05, 0.2)
        total_pct = train_pct + valid_pct + test_pct
        if total_pct > 1:
            g.exit_app(
                "SharedTraining.__split_dataset(): dataset total percent must <= 1"
            )

        # dataset_split_seed keeps train/valid/test unchanged everytime
        patient_list = g.get_sub_folders(
            folder_path=g.DATASET_FOLDER,
            key_word="GTVt",
            shuffle=True,
            seed=dataset_split_seed,
        )

        train_num = int(len(patient_list) * train_pct)
        valid_num = int(len(patient_list) * valid_pct)
        if total_pct == 1:
            test_num = len(patient_list) - train_num - valid_num
        elif total_pct < 1:
            test_num = int(len(patient_list) * test_pct)

        json_data = NestedDict()
        json_data["train.patient.list"] = patient_list[:train_num]
        patient_list = patient_list[train_num:]
        json_data["valid.patient.list"] = patient_list[:valid_num]
        patient_list = patient_list[valid_num:]
        json_data["test.patient.list"] = patient_list[:test_num]

        g.save_json(json_data, g.DATASET_SPLITTING_JSON)

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
        group_start_time: str,
        train_remark: str,
        debug_mode: bool,
        full_hyper_dict: dict,
        cur_hyper_dict: dict,
    ):
        train_id = group_start_time
        cur_start_time = self._init_start_time()
        train_id += "_" + cur_start_time

        if debug_mode:
            train_id += "_debug.mode.delete.this"

        if train_remark != "" and train_remark is not None:
            while train_remark.startswith("_"):
                train_remark = train_remark[1:]
            while train_remark.endswith("_"):
                train_remark = train_remark[:-1]
            train_id += "_" + train_remark

        # add important hyper param (that need to be compared) to train_id
        for key in full_hyper_dict:
            if len(full_hyper_dict[key]) > 1:
                # replace "_" with "." in key name
                train_id += "_" + key.replace("_", ".")
                train_id += "=" + str(cur_hyper_dict[key]).replace("_", ".")
        return train_id

    def _optimize_batch_size(self, dataset: Union[BaselineDataSet, IDLDataSet]):
        dataset_len = dataset.__len__()
        if dataset_len > self._batch_size_actual:
            batch_size = math.ceil(
                dataset_len / (math.ceil(dataset_len / self._batch_size_actual))
            )
        else:
            batch_size = dataset_len  # self._batch_size_actual
        return batch_size

    def _inference(
        self,
        patient: str,
        imgs_save_folder: str = None,
        save_pred_only: bool = False,
    ):
        score_dict = NestedDict()

        self._cnn.eval()  # disable dropout / batch nomalize
        with torch.no_grad():

            inputs, labels = BaselineDataSet(patient_list=[patient]).__getitem__(0)
            inputs = torch.unsqueeze(inputs.to(g.DEVICE), dim=0)
            labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
            outputs = self._cnn.forward(inputs)

        inputs = torch.squeeze(inputs, dim=0)
        labels = torch.squeeze(labels, dim=0)
        outputs = torch.squeeze(outputs, dim=0)

        for score_type in ["dsc", "msd", "hd95"]:
            score_dict["gtvt"][score_type] = self._score_funcs[score_type](
                outputs[0], labels[0]
            )
            score_dict["gtvn"][score_type] = self._score_funcs[score_type](
                outputs[1], labels[1]
            )

        # save images
        if imgs_save_folder is not None:
            g.create_folder(imgs_save_folder)

            # get pred
            pred = NestedDict()
            pred["gtvt"] = outputs[0].cpu().numpy()
            pred["gtvn"] = outputs[1].cpu().numpy()

            # save pred
            for i in ["gtvt", "gtvn"]:
                g.save_nii(
                    np_data=pred[i],
                    save_path=os.path.join(imgs_save_folder, "pred_{}.nii".format(i)),
                    spacing=g.NII_SPACING,
                )

            if not save_pred_only:
                imgs = NestedDict()
                imgs["ct"] = inputs[0].cpu().numpy()
                imgs["pt"] = inputs[1].cpu().numpy()
                imgs["mrt1"] = inputs[2].cpu().numpy()
                imgs["mrt2"] = inputs[3].cpu().numpy()
                imgs["label_gtvt"] = labels[0].cpu().numpy()
                imgs["label_gtvn"] = labels[1].cpu().numpy()

                # save imgs and labels
                for i in ["ct", "pt", "mrt1", "mrt2", "label_gtvt", "label_gtvn"]:
                    g.save_nii(
                        np_data=imgs[i],
                        save_path=os.path.join(imgs_save_folder, "{}.nii".format(i)),
                        spacing=g.NII_SPACING,
                    )

        return score_dict

    # protected function
    def _init_start_time(self) -> str:
        start_time = str(datetime.now().replace(microsecond=0))
        start_time = start_time.replace(":", ".")
        start_time = start_time.replace("-", ".")
        start_time = start_time.replace(" ", ".")
        return start_time

    def _load_full_hyper(self, hyper_json_path: str) -> dict:
        # load hyper param
        full_hyper_dict = g.load_json(hyper_json_path)

        # make sure all values of hyper dict are "list" type
        for i in full_hyper_dict:
            if not isinstance(full_hyper_dict[i], list):
                full_hyper_dict[i] = [full_hyper_dict[i]]

        return full_hyper_dict
