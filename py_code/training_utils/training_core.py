import math
import os
import random
from collections import OrderedDict
from itertools import product
from pathlib import Path

import global_utils.global_core as g
import torch
from dataset_utils.dataset_baseline import DataSetBaseline
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetPart, DatasetVer, ErrMsg, Metric, Stat
from metric_utils.metric_func import MetricFunction
from numpy import ndarray
from torch import optim
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unet_pp_slim import UNetPPSlim
from unet_slim import UNetSlim


class ObsStudyProgress:
    def __init__(self):
        self.cur_step = None
        self.total_step = None
        self.progress_signal = None

    def emit_signal(self):
        if (
            self.progress_signal is not None
            and self.cur_step is not None
            and self.total_step is not None
        ):
            self.progress_signal.emit(self.cur_step / self.total_step)


class TrainingCore:
    def __init__(self):
        # self._timer = Timer()
        self._obs_study_progress = None

    def _load_patients(
        self,
        dataset_ver: str,
        fold: int = None,  # fold=None means no validation set, but use all folds as training set
        debug_mode: bool = False,
    ):
        patients = Dict()
        dataset_split = g.load_json(g.DATASET_SPLIT_PATH[dataset_ver])

        if dataset_ver in [DatasetVer.AU_EXT, DatasetVer.OBS_STUDY]:
            patients[DatasetPart.TEST] = List(dataset_split[DatasetPart.TEST])

        else:
            # calculate fold count
            fold_count = 0
            for key_name in dataset_split.keys():
                if "fold." in key_name:
                    fold_count += 1
            if fold_count != g.DATASET_FOLDS[dataset_ver]:
                dataset_split = self.__kfolds_split(dataset_ver)
            # test set
            patients[DatasetPart.TEST] = List(dataset_split[DatasetPart.TEST])
            # valid set
            patients[DatasetPart.VALID] = List(dataset_split["fold.{}".format(fold)])
            # train set
            patients[DatasetPart.TRAIN] = List()
            for key_name in dataset_split.keys():
                if "fold." in key_name and key_name != "fold.{}".format(fold):
                    patients[DatasetPart.TRAIN] += List(dataset_split[key_name])

        if debug_mode:
            for dataset_part in patients.keys():
                # keep 2 patients to test median score calculation
                # keep 1 patient for faster debugging
                patients[dataset_part] = patients[dataset_part][:2]

        return patients

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_hyper_new_cnn(self, hyper: Dict, in_chan: int, out_chan: int):
        # cnn architecture
        if hyper["cnn"] == "unet.pp.slim":
            cnn = UNetPPSlim
        elif hyper["cnn"] == "unet.slim":
            cnn = UNetSlim
        else:
            g.error_exit("Incorrect hyper[cnn] value!")
        hyper["cnn"] = cnn(
            in_chan=in_chan,
            out_chan=out_chan,
            dataset_ver=hyper["dataset.ver"],
            dropout=hyper["dropout"],
        )
        # set multi-GPU
        if g.used_gpu_count() > 1:
            hyper["cnn"] = DataParallel(hyper["cnn"])
        # to gpu (if gpu available)
        hyper["cnn"] = hyper["cnn"].to(g.DEVICE)

    def _load_exist_cnn(self, cnn_path: str):
        cnn = torch.load(cnn_path)
        if g.used_gpu_count() > 1:
            cnn = DataParallel(cnn)
        cnn = cnn.to(g.DEVICE)
        return cnn

    def _load_metric_funcs(self) -> Dict:
        metric_funcs = Dict()
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            metric_funcs[metric] = MetricFunction(metric)
            # following line will cause bug, cant figure out why:
            # if g.used_gpu_count() > 1:
            #     metric_funcs[metric] = DataParallel(metric_funcs[metric])
            metric_funcs[metric] = metric_funcs[metric].to(g.DEVICE)
        return metric_funcs

    def _load_hyper_dataset_version(self, hyper: Dict, idl_baseline_id: str):
        # baseline
        if idl_baseline_id is None:
            hyper["dataset.ver"] = self._is_valid_dataset_version(
                dataset_ver=hyper["dataset.ver"]
            )
        # idl
        else:
            baseline_dir = os.path.join(
                g.TRAIN_RESULTS_DIR, idl_baseline_id, "baseline"
            )
            baseline_fold_dir = g.get_sub_dirs(
                baseline_dir, key_word="fold=", full_path=True
            )[0]
            baseline_dataset_ver = g.load_json(
                os.path.join(baseline_fold_dir, "hyper.json")
            )["dataset.ver"]
            hyper["dataset.ver"] = self._is_valid_dataset_version(
                dataset_ver=hyper["dataset.ver"],
                origin_dataset_ver=baseline_dataset_ver,
            )

    def _load_hyper(self, hyper: Dict) -> None:
        # device name
        if g.used_gpu_count() < 1:
            hyper["device"] = "cpu"
        else:
            hyper["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]

        # dropout
        hyper["dropout"] = g.clamp_value(hyper["dropout"], (0.0, 1.0))

        # batch size
        hyper["batch.size"] = g.clamp_value(hyper["batch.size"], (1, None))
        if g.used_gpu_count() > 1:
            hyper["batch.size.actual"] = hyper["batch.size"] * g.used_gpu_count()
        else:
            hyper["batch.size.actual"] = hyper["batch.size"]

        # = 1 will cause error
        hyper["lr.decay.factor"] = g.clamp_value(
            hyper["lr.decay.factor"], (g.EPS, 1 - g.EPS)
        )

        # augment methods
        hyper["augment.methods"] = List(hyper["augment.methods"])

        # augment lower/upper limit
        hyper["augment.max"] = g.clamp_value(
            hyper["augment.max"], (1, len(hyper["augment.methods"]))
        )
        hyper["augment.min"] = g.clamp_value(
            hyper["augment.min"], (1, hyper["augment.max"])
        )

        # loss function parameters
        hyper["loss.weight"] = g.clamp_value(hyper["loss.weight"], (0.0, 1.0))
        hyper["loss.delta"] = g.clamp_value(hyper["loss.delta"], (0.0, 1.0))

    def _load_hyper_optim_and_scheduler(self, hyper: Dict, lr: float = None):
        if lr is None:
            lr = hyper["lr.actual"]

        # optimizer (no need to move to cuda)
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

        for key_name in hyper.keys():
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
                elif isinstance(cnn, UNetSlim):
                    simple_hyper[key_name] = "unet.slim"
                else:
                    g.error_exit("Incorrect cnn type!")

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

    # split dataset into k folds and save dict into json file
    # this function keep test set unchanged
    def __kfolds_split(self, dataset_ver: str) -> Dict:
        dataset_split = g.load_json(g.DATASET_SPLIT_PATH[dataset_ver])
        train_patients = List()

        for key_name in dataset_split.keys():
            if "fold." in key_name:
                train_patients += List(dataset_split[key_name])
                dataset_split.pop(key_name)

        random.shuffle(train_patients)

        fold_len = round(len(train_patients) / g.DATASET_FOLDS[dataset_ver])
        for fold in range(1, g.DATASET_FOLDS[dataset_ver] + 1):
            if fold == g.DATASET_FOLDS[dataset_ver]:
                dataset_split["fold.{}".format(fold)] = train_patients.to_str()
            else:
                dataset_split["fold.{}".format(fold)] = train_patients[
                    :fold_len
                ].to_str()
                train_patients = train_patients[fold_len:]

        g.save_json(dataset_split, g.DATASET_SPLIT_PATH[dataset_ver])
        return dataset_split

    def _save_cnn(self, hyper: Dict, save_path: str):
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        if g.used_gpu_count() > 1:
            torch.save(hyper["cnn"].module, save_path)
        else:
            torch.save(hyper["cnn"], save_path)

    # train_id = start_time + train_remark
    def _init_train_id(
        self, hyper: Dict, hyper_json_path: str, train_remark: str, debug_mode: bool
    ) -> str:
        train_id = g.get_cur_time_str()

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

    # evenly distribute the data into each batch
    def _optimize_batch_size(self, dataset, hyper: Dict):
        if dataset.__len__() > hyper["batch.size"]:
            hyper["batch.size"] = math.ceil(
                dataset.__len__() / (math.ceil(dataset.__len__() / hyper["batch.size"]))
            )
        else:
            hyper["batch.size"] = dataset.__len__()

    def _load_hyper_series_from_json(self, path: str) -> List:
        hyper_series_list = List()
        origin_hyper_dict = g.load_json(path)
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
            hyper_series_list.append(cur_hyper)

        return hyper_series_list

    # find train result directory full path using train_id
    # for baseline, it will return the "baseline_xxxx/baseline" dir
    def _find_train_dir(self, train_id: str) -> str:
        baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id, "baseline")

        # train id is a baseline
        if os.path.exists(baseline_dir):
            return baseline_dir

        # train id is a iDL
        else:
            for baseline_dir in g.get_sub_dirs(g.TRAIN_RESULTS_DIR, full_path=True):
                for train_dir in g.get_sub_dirs(baseline_dir, full_path=True):
                    if Path(train_dir).name == train_id:
                        return train_dir
            # cant find train_dir
            return None

    def _inference_single_patient(
        self,
        patient: str,
        cnn,
        dataset_ver: str,
        no_pt: bool,
        no_mr: bool,
        metric_funcs: Dict = None,
        idl_gtvt_label_masked_by_selected_slices: ndarray = None,  # only for idl.gtvt post processing
        idl_gtvn_geodesic_distance: bool = True,  # only for idl.gtvn
        obs_gtvn_clicks: ndarray = None,  # only for observer study
    ) -> Dict:

        # (1)outputs structure for sigle-observer datasets:
        # [gtv]->["dsc/msd/hd95/label/pred/clicks/distance.map"]

        # (2)outputs structure for MDA dataset:
        # [gtv]->["label/clicks/distance.map"]->["observer1/2/3"]
        # [gtv]->["dsc/msd/hd95"]->["observer1/2/3/iov"]
        outputs = Dict()

        # load dataset
        dataset = self._inference_single_patient_load_dataset(
            patient=patient,
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            no_mr=no_mr,
            idl_gtvn_geodesic_distance=idl_gtvn_geodesic_distance,
            obs_gtvn_clicks=obs_gtvn_clicks,  # only for observer study gtvn
        )

        input_imgs = None
        img_shape = None

        # get items from dataset
        # augment was set to None when creating dataset,
        # so there is no data augmentation in get_item() function
        dataset_item = dataset.get_item(patient)
        if dataset_item is None:
            return None

        # get input images
        if input_imgs is None:
            input_imgs = dataset_item["input.imgs"]
            # add "batch" (c/d/h/w -> b/c/d/h/w)
            input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)

        # get img shape
        if img_shape is None:
            img_shape = dataset_item["shape"]

        # record labels of current observer into outputs dict
        self._inference_single_patient_record_labels(
            outputs=outputs,
            dataset_item=dataset_item,
        )

        # record gtvn clicks of current observer into outputs dict
        self._inference_single_patient_record_gtvn_clicks(
            outputs=outputs,
            dataset_item=dataset_item,
        )

        # record gtvn distance map into outputs
        self._inference_single_patient_record_gtvn_distance_map(
            outputs=outputs,
            input_imgs=input_imgs,
            img_shape=img_shape,
        )

        # idl progress INFERENCE_LOAD_IMG
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step += (
                self._obs_study_progress.step.INFERENCE_LOAD_IMG
            )
            self._obs_study_progress.emit_signal()

        # get pred using inputs and cnn
        cnn.eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            preds = cnn.forward(input_imgs)
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        preds = torch.squeeze(preds, dim=0).cpu().numpy()

        # observer study progress INFERENCE_FORWARD
        if self._obs_study_progress is not None:
            self._obs_study_progress.cur_step += (
                self._obs_study_progress.step.INFERENCE_FORWARD
            )
            self._obs_study_progress.emit_signal()

        # record preds into outputs
        self._inference_single_patient_record_preds(
            outputs=outputs,
            preds=preds,
            img_shape=img_shape,
        )

        # post processing (before calculate metric)
        # remove connected_components has no overlap with delineated slices
        self._inference_single_patient_gtvt_post_process(
            outputs=outputs,
            idl_gtvt_label_masked_by_selected_slices=idl_gtvt_label_masked_by_selected_slices,
        )

        self._inference_single_patient_gtvn_post_process(outputs)

        # calculate metrics
        if metric_funcs is not None and self._obs_study_progress is None:
            for gtv in outputs.keys():
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    outputs[gtv][metric] = metric_funcs[metric](
                        outputs[gtv]["pred"],
                        outputs[gtv]["label"],
                    )

        return outputs

    def _inference_single_patient_load_dataset(
        self,
        patient: str,
        dataset_ver: str,
        no_pt: bool,
        no_mr: bool,
        *args,
        **kwargs,
    ):
        return DataSetBaseline(
            patients=[patient],
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            no_mr=no_mr,
            augment=None,
        )

    def _inference_single_patient_record_labels(self, *args, **kwargs):
        pass

    def _inference_single_patient_record_gtvn_clicks(self, *args, **kwargs):
        pass

    def _inference_single_patient_record_preds(self, *args, **kwargs):
        pass

    def _inference_single_patient_record_gtvn_distance_map(self, *args, **kwargs):
        pass

    def _inference_single_patient_gtvn_post_process(self, *args, **kwargs):
        pass

    def _inference_single_patient_gtvt_post_process(self, *args, **kwargs):
        pass

    # make this function protected, idl will use it
    def _is_valid_baseline_id(self, baseline_id: str):
        if not baseline_id.startswith("baseline_"):
            g.error_exit("'baseline_id' must start with 'baseline_'!")

        if not os.path.exists(os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)):
            g.error_exit("'baseline_id' does not exist!")

    def _is_valid_dataset_version(
        self,
        dataset_ver,
        origin_dataset_ver=None,
    ):
        # origin_dataset_ver represent "baseline dataset version" in iDL
        # or represent "training dataset version" in inference

        if origin_dataset_ver is not None:
            # copy origin_dataset_ver if dataset_ver is None
            if dataset_ver is None:
                dataset_ver = origin_dataset_ver

            if origin_dataset_ver not in [
                DatasetVer.AU,
                DatasetVer.MDA,
                DatasetVer.NKI,
                DatasetVer.HECKTOR,
            ]:
                g.error_exit(ErrMsg.DATASET_VER_INVALID)

            elif origin_dataset_ver == DatasetVer.MDA and dataset_ver != DatasetVer.MDA:
                g.error_exit(ErrMsg.DATASET_VER_INVALID)

            elif origin_dataset_ver == DatasetVer.NKI and dataset_ver != DatasetVer.NKI:
                g.error_exit(ErrMsg.DATASET_VER_INVALID)

            elif (
                origin_dataset_ver == DatasetVer.HECKTOR
                and dataset_ver != DatasetVer.HECKTOR
            ):
                g.error_exit(ErrMsg.DATASET_VER_INVALID)

        elif dataset_ver not in [
            DatasetVer.AU,
            DatasetVer.AU_EXT,
            DatasetVer.OBS_STUDY,
            DatasetVer.MDA,
            DatasetVer.NKI,
            DatasetVer.HECKTOR,
        ]:
            g.error_exit(ErrMsg.DATASET_VER_INVALID)

        return dataset_ver

    def _find_best_cnn_in_folds(self, baseline_id: str) -> str:
        self._is_valid_baseline_id(baseline_id)

        scores = Dict()

        baseline_dir = self._find_train_dir(baseline_id)

        fold_dirs = g.get_sub_dirs(
            input_dir=baseline_dir,
            key_word="fold=",
            full_path=True,
        )
        for fold_dir in fold_dirs:
            dataset_ver = g.load_json(os.path.join(fold_dir, "hyper.json"))[
                "dataset.ver"
            ]

            fold = Path(fold_dir).name
            epoch_dir = g.get_sub_dirs(fold_dir, key_word="epoch=", full_path=True)[0]
            epoch_scores = g.load_json(
                os.path.join(
                    epoch_dir,
                    "inference_{}_valid.json".format(dataset_ver),
                )
            )
            for stat in [Stat.MEDIAN, Stat.AVG]:
                scores[fold][stat] = epoch_scores[stat]

        for stat in [Stat.MEDIAN, Stat.AVG]:
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    # create a tmp list to sort
                    list_to_sort = List()
                    # add elements into the list
                    for epoch in scores.keys():
                        list_to_sort.append(scores[epoch][stat][gtv][metric])
                    # sort the list
                    if metric == Metric.DSC:
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)
                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        new_value = list_to_sort.index(scores[epoch][stat][gtv][metric])
                        # if metric == Metric.DSC:
                        #     new_value *= 2
                        scores[epoch][stat][gtv][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stat in [Stat.AVG, Stat.MEDIAN]:
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                        evaluation[epoch] += scores[epoch][stat][gtv][metric]

        best_fold = evaluation.key_with_max_value()
        best_epoch_dir = g.get_sub_dirs(
            os.path.join(baseline_dir, best_fold), key_word="epoch=", full_path=True
        )[0]
        best_cnn_path = g.get_sub_files(best_epoch_dir, key_word=".pt", full_path=True)[
            0
        ]
        return best_cnn_path
