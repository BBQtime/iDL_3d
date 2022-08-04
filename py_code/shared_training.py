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
from unet_2d import UNet2D
from unet_pp_2d import UNetPP2D
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
            for dim in ["2d", "3d"]:
                self._score_funcs[score_type][dim] = crit.ScoreFunction(
                    score_type=score_type, dim=dim
                ).to(g.DEVICE)

    # can be shared with baseline_visualize
    def _load_batch_size(self, hyper_dict: dict) -> tuple[int, int]:
        batch_size = int(hyper_dict["batch.size"])
        batch_size = g.check_limit(batch_size, 1, None)

        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            batch_size_actual = batch_size * used_gpu_count
        else:
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
            dim="2d",
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
            if cnn_name == "unet":
                cnn = UNet2D(dropout=self._dropout).to(g.DEVICE)
            elif cnn_name == "unet++":
                cnn = UNetPP2D(dropout=self._dropout).to(g.DEVICE)

        # exist cnn
        else:
            # load state dict only
            if g.CNN_STATE_DICT_ONLY:
                if cnn_name == "unet":
                    cnn = UNet2D(dropout=self._dropout).to(g.DEVICE)
                elif cnn_name == "unet++":
                    cnn = UNetPP2D(dropout=self._dropout).to(g.DEVICE)
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
        if isinstance(cnn, UNet2D):
            return "unet"
        elif isinstance(cnn, UNetPP2D):
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
            g.DATASET_FOLDER, shuffle=True, seed=dataset_split_seed
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

    def _test_patients(
        self,
        patient_list: list,
        cnn: Union[UNet2D, UNetPP2D],
        imgs_save_folder: str = None,
        save_pred_only: bool = False,
        show_tqdm_bar: bool = False,
    ):
        cnn.eval()  # disable dropout / batch nomalize
        with torch.no_grad():

            all_patient_results = NestedDict()

            # tqdm bar
            if show_tqdm_bar:
                tqdm_patient_list = tqdm(patient_list)
            else:
                tqdm_patient_list = patient_list

            # patient loop
            for cur_patient in tqdm_patient_list:
                cur_patient_set = BaselineDataSet(patient_list=[cur_patient])
                batch_size = self._optimize_batch_size(cur_patient_set)

                # dataloader (only current patient)
                cur_patient_loader = DataLoader(
                    dataset=cur_patient_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=g.NUM_WORKERS,
                )

                # return img type
                if imgs_save_folder is None:
                    return_img_type = None
                elif save_pred_only:
                    return_img_type = "pred"
                else:
                    return_img_type = "all"

                # calculate result
                # test_results["score"]["patient"]["dsc/msd/hd95"]["2d/3d"]
                # test_results# test_results["ct/pet/mrt1/mrt2"]
                cur_patient_results = self.__test_single_patient(
                    data_loader=cur_patient_loader,
                    cnn=cnn,
                    return_img_type=return_img_type,
                )

                # save cur patient imgs
                if imgs_save_folder is not None:
                    if len(patient_list) > 1:
                        cur_patient_folder = os.path.join(
                            imgs_save_folder, "patient=" + cur_patient
                        )
                    else:
                        cur_patient_folder = imgs_save_folder
                    g.create_folder(cur_patient_folder)

                    # save ct/pet/mr1/mr2/label
                    if not save_pred_only:
                        for i in ["ct", "pet", "mrt1", "mrt2", "label"]:
                            g.save_nii(
                                np_data=cur_patient_results[i],
                                save_path=os.path.join(cur_patient_folder, i + ".nii"),
                                spacing=g.NII_SPACING,
                            )
                    # save pred
                    g.save_nii(
                        np_data=cur_patient_results["pred"],
                        save_path=os.path.join(cur_patient_folder, "pred.nii"),
                        spacing=g.NII_SPACING,
                    )
                all_patient_results[cur_patient] = cur_patient_results["score"]

            return all_patient_results

    def __test_single_patient(
        self,
        data_loader: DataLoader,
        cnn: Union[UNet2D, UNetPP2D],
        return_img_type: str = None,  # all/pred/None
    ):
        cnn.eval()
        with torch.no_grad():

            # init test_results dict
            test_results = NestedDict()
            if return_img_type is not None:
                test_results["pred"] = None
                if return_img_type == "all":
                    for i in ["ct", "pet", "mrt1", "mrt2", "label"]:
                        test_results[i] = None

            patient_slice_mapping = data_loader.dataset.patient_slice_mapping

            inputs = None
            labels = None
            outputs = None
            # go through data loader
            for cur_inputs, cur_labels in data_loader:
                # this step is time consuming
                cur_outputs = cnn(cur_inputs.to(g.DEVICE))

                # concat inputs
                if inputs is None:
                    inputs = cur_inputs
                else:
                    inputs = torch.cat([inputs, cur_inputs], dim=0)

                # concat labels
                if labels is None:
                    labels = cur_labels
                else:
                    labels = torch.cat([labels, cur_labels], dim=0)

                # concat outputs
                if outputs is None:
                    outputs = cur_outputs.cpu()
                else:
                    outputs = torch.cat([outputs, cur_outputs.cpu()], dim=0)

            inputs = inputs.to(g.DEVICE)
            labels = labels.to(g.DEVICE)
            outputs = outputs.to(g.DEVICE)

            for score_type in ["dsc", "msd", "hd95"]:
                for dim in ["2d", "3d"]:

                    # 2d score is a list
                    if dim == "2d":
                        score_list = self._score_funcs[score_type][dim](outputs, labels)

                        mapping_idx = 0  # index of "patient_slice_mapping"
                        # calculate sum score to get avg.2d.score
                        sum_score = 0
                        score_count = 0

                        for cur_score in score_list:
                            # "patient_slice_mapping" format:
                            # [[patient_id, slice_id], [patient_id, slice_id]]
                            cur_slice = patient_slice_mapping[mapping_idx][1]
                            test_results["score"][score_type][dim][
                                cur_slice
                            ] = cur_score
                            mapping_idx += 1

                            # calculate sum score to get avg.2d.score
                            if g.is_number(cur_score):
                                sum_score += cur_score
                                score_count += 1
                            elif cur_score == "no.pred" or cur_score == "no.label":
                                if score_type == "dsc":
                                    sum_score += 0
                                elif score_type == "msd":
                                    sum_score += g.IMG_SIZE
                                elif score_type == "hd95":
                                    sum_score += g.IMG_SIZE
                                score_count += 1
                            else:  # cur_score == "empty"
                                pass

                        if score_count == 0:
                            test_results["score"][score_type]["2d.avg"] = "empty"
                        else:
                            test_results["score"][score_type]["2d.avg"] = (
                                sum_score / score_count
                            )

                    # 3d score is a single value
                    else:
                        score = self._score_funcs[score_type][dim](outputs, labels)
                        test_results["score"][score_type][dim] = score

            #  record images
            if return_img_type is not None:
                # prediction
                test_results["pred"] = outputs[:, 0, :, :].cpu().numpy()

                if return_img_type == "all":
                    # ct/pet/mrt1/mrt2
                    channel = 0
                    for i in ["ct", "pet", "mrt1", "mrt2"]:
                        test_results[i] = inputs[:, channel, :, :].cpu().numpy()
                        channel += 1

                    # label
                    test_results["label"] = labels[:, 0, :, :].cpu().numpy()

            return test_results

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
