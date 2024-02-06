import math
import os
import random
from collections import OrderedDict
from itertools import product
from pathlib import Path

import torch
from custom import GPU, Debug, Dict, Dir
from custom import Global as g
from custom import Img, Json, List, Timer, Value
from dataset_baseline import DataSetBaseline
from numpy import ndarray
from segment_metric import SegmentationMetric
from str_lib import DatasetPart, DatasetVer, Metric
from torch import Tensor, optim
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unet_pp_slim import UNetPPSlim
from unet_slim import UNetSlim


class RealIDLProgress:
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
        self._timer = Timer()
        self._idl_progress = None

    def _load_patients(
        self,
        dataset_ver: str,
        fold: int = None,  # fold=None means no validation set, but use all folds as training set
        debug_mode: bool = False,
    ):
        dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH[dataset_ver])

        # calculate fold count
        fold_count = 0
        for key_name in dataset_split.keys():
            if "fold." in key_name:
                fold_count += 1

        if fold_count != g.DATASET_FOLDS:
            dataset_split = self.__split_dataset()

        patients = Dict()
        # test set
        for key_name in [
            DatasetPart.TEST_INTER,
            DatasetPart.TEST_EXTER,
            DatasetPart.TEST,
        ]:
            patients[key_name] = List(dataset_split[key_name])
        # valid set
        patients[DatasetPart.VALID] = List(dataset_split["fold.{}".format(fold)])
        # train set
        patients[DatasetPart.TRAIN] = List()
        for key_name in dataset_split.keys():
            if "fold." in key_name and key_name != "fold.{}".format(fold):
                patients[DatasetPart.TRAIN] += List(dataset_split[key_name])

        if debug_mode:
            for key_name in patients.keys():
                # keep 2 patients to test median score calculation
                # keep 1 patient for faster debugging
                patients[key_name] = patients[key_name][:2]

        return patients

    # if float64 needed, use: "cnn.to(torch.double)"
    def _load_hyper_new_cnn(self, hyper: Dict, in_chan: int, out_chan: int):
        # cnn architecture
        if hyper["cnn"] == "unet.pp.slim":
            cnn = UNetPPSlim
        elif hyper["cnn"] == "unet.slim":
            cnn = UNetSlim
        else:
            Debug.error_exit("Incorrect hyper[cnn] value!")
        hyper["cnn"] = cnn(
            in_chan=in_chan,
            out_chan=out_chan,
            dataset_ver=hyper["dataset.ver"],
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

    def _load_segment_metrics(self, dataset_ver: str) -> Dict:
        segment_metrics = Dict()
        for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
            segment_metrics[metric] = SegmentationMetric(
                metric=metric, dataset_ver=dataset_ver
            )
            # following line will cause bug, cant figure out why:
            # if GPU.used_count() > 1:
            #     segment_metrics[metric] = DataParallel(segment_metrics[metric])
            segment_metrics[metric] = segment_metrics[metric].to(g.DEVICE)
        return segment_metrics

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
            baseline_fold_dir = Dir.get_sub_dirs(
                baseline_dir, key_word="fold=", full_path=True
            )[0]
            baseline_dataset_ver = Json.load(
                os.path.join(baseline_fold_dir, "hyper.json")
            )["dataset.ver"]
            hyper["dataset.ver"] = self._is_valid_dataset_version(
                dataset_ver=hyper["dataset.ver"],
                origin_dataset_ver=baseline_dataset_ver,
            )

    def _load_hyper(self, hyper: Dict) -> None:
        # device name
        if GPU.used_count() < 1:
            hyper["device"] = "cpu"
        else:
            hyper["device"] = "gpu:" + os.environ["CUDA_VISIBLE_DEVICES"]

        # dropout
        hyper["dropout"] = Value.limit_range(hyper["dropout"], (0.0, 1.0))

        # batch size
        hyper["batch.size"] = Value.limit_range(hyper["batch.size"], (1, None))
        if GPU.used_count() > 1:
            hyper["batch.size.actual"] = hyper["batch.size"] * GPU.used_count()
        else:
            hyper["batch.size.actual"] = hyper["batch.size"]

        # = 1 will cause error
        hyper["lr.decay.factor"] = Value.limit_range(
            hyper["lr.decay.factor"], (Value.EPS, 1 - Value.EPS)
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
                    Debug.error_exit("Incorrect cnn type!")

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
    def __split_dataset(self) -> Dict:
        dataset_split = Json.load(g.DATASET_SPLIT_JSON_PATH)
        train_patients = List()

        for key_name in dataset_split.keys():
            if "fold." in key_name:
                train_patients += List(dataset_split[key_name])
                dataset_split.pop(key_name)

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
        self, hyper: Dict, hyper_json_path: str, train_remark: str, debug_mode: bool
    ) -> str:
        train_id = Timer.cur_time_str()

        if debug_mode:
            train_id += "_"
            train_id += Debug.DELETE_FLAG

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

    def _load_hyper_series_from_json(self, path: str) -> List:
        hyper_series_list = List()
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
            for baseline_dir in Dir.get_sub_dirs(g.TRAIN_RESULTS_DIR, full_path=True):
                for train_dir in Dir.get_sub_dirs(baseline_dir, full_path=True):
                    if Path(train_dir).name == train_id:
                        return train_dir
            # cant find train_dir
            return None

    def _inference_single_patient(
        self,
        patient: str,
        cnn,
        dataset_ver: str,
        dataset_part: str,
        no_pt: str,
        segment_metrics: Dict = None,
        idl_gtvn_baseline_id: str = None,  # only for idl.gtvn
        idl_gtvn_clicks: ndarray = None,  # only for idl.gtvn
        idl_gtvt_masked_label: ndarray = None,  # only for idl.gtvt post processing
    ) -> Dict:
        # load dataset
        dataset = self._inference_single_patient_load_dataset(
            patient=patient,
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            idl_gtvn_baseline_id=idl_gtvn_baseline_id,
            idl_gtvn_clicks=idl_gtvn_clicks,
        )

        # load labels
        labels = Img.load_labels(
            dataset_dir=g.DATASET_DIR[dataset_ver], patient=patient
        )
        # outputs structure: ["gtvs/gtvt/gtvn"]["label/pred/clicks/distance.map"]
        outputs = self._inference_single_patient_record_labels(labels)

        # get items from dataset
        item = dataset.get_item(patient)
        input_imgs = item[0]
        labels = item[1]
        idl_gtvn_clicks = self._inference_single_patient_get_gtvn_clicks(item)

        # add "batch" (c/d/h/w -> b/c/d/h/w)
        input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)
        labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)

        # idl progress INFERENCE_LOAD_IMG
        self._timer.cal_duration("INFERENCE_LOAD_IMG")
        if self._idl_progress is not None:
            self._idl_progress.cur_step += self._idl_progress.step.INFERENCE_LOAD_IMG
            self._idl_progress.emit_signal()

        # get predictions from cnn
        cnn.eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            preds = cnn.forward(input_imgs)
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        preds = torch.squeeze(preds, dim=0).cpu().numpy()

        # idl progress INFERENCE_FORWARD
        self._timer.cal_duration("INFERENCE_FORWARD")
        if self._idl_progress is not None:
            self._idl_progress.cur_step += self._idl_progress.step.INFERENCE_FORWARD
            self._idl_progress.emit_signal()

        # record img into outputs
        self._inference_single_patient_record_outputs(
            outputs=outputs,
            preds=preds,
            input_imgs=input_imgs,
            idl_gtvn_clicks=idl_gtvn_clicks,
        )

        # pad and crop all imgs to original size
        for gtv in outputs.keys():
            for i in outputs[gtv].keys():
                outputs[gtv][i] = Img.central_pad_and_crop(
                    outputs[gtv][i], outputs[gtv]["label"].shape
                )

        # post processing (after pad and crop, before calculate scores)
        self._inference_single_patient_gtvt_post_process(
            outputs=outputs, idl_gtvt_masked_label=idl_gtvt_masked_label
        )
        self._inference_single_patient_gtvn_post_process(outputs)

        # calculate scores of current patient
        if dataset_part != DatasetPart.TRAIN and self._idl_progress is None:
            for gtv in outputs.keys():
                for metric in [Metric.DSC, Metric.MSD, Metric.HD95]:
                    outputs[gtv][metric] = segment_metrics[metric](
                        outputs[gtv]["pred"],
                        outputs[gtv]["label"],
                    )

        self._timer.cal_duration("INFERENCE_CAL_SCORES")

        return outputs

    def _inference_single_patient_load_dataset(
        self,
        patient: str,
        dataset_ver: str,
        no_pt: bool,
        idl_gtvn_baseline_id: str = None,
        idl_gtvn_clicks: ndarray = None,
    ):
        return DataSetBaseline(
            patients=[patient],
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            augment=None,
        )

    def _inference_single_patient_record_labels(self, labels: Dict, outputs: Dict):
        pass

    def _inference_single_patient_get_gtvn_clicks(self, item: list):
        return None

    def _inference_single_patient_record_outputs(
        self, outputs: Dict, preds: Dict, input_imgs: Tensor, idl_gtvn_clicks: Tensor
    ):
        pass

    def _inference_single_patient_gtvn_post_process(self, outputs: Dict):
        pass

    def _inference_single_patient_gtvt_post_process(
        self, outputs: Dict, idl_gtvt_masked_label: ndarray
    ):
        pass

    # make this function protected, idl will use it
    def _is_valid_baseline_id(self, baseline_id: str):
        if not baseline_id.startswith("baseline_"):
            Debug.error_exit("'baseline_id' must start with 'baseline_'!")

        if not os.path.exists(os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)):
            Debug.error_exit("'baseline_id' does not exist!")

    def _is_valid_dataset_version(
        self,
        dataset_ver,
        origin_dataset_ver=None,  # this is for inference and idl
    ):
        if origin_dataset_ver is not None:
            # copy origin_dataset_ver if dataset_ver is None
            if dataset_ver is None:
                dataset_ver = origin_dataset_ver

            if origin_dataset_ver == DatasetVer.MDA:
                if dataset_ver != DatasetVer.MDA:
                    Debug.error_exit(
                        "Due to existing train info, 'dataset_ver' is restricted to 'MDA' only!"
                    )

            elif origin_dataset_ver == DatasetVer.AU_1MM:
                if dataset_ver == DatasetVer.AU_3MM:
                    Debug.error_exit(
                        "Due to existing train info, 'dataset_ver' can not be 'AU.3mm'!"
                    )

            elif origin_dataset_ver == DatasetVer.AU_3MM:
                if dataset_ver != DatasetVer.AU_3MM:
                    Debug.error_exit(
                        "Due to existing train info, 'dataset_ver' is restricted to 'AU.3mm' only!"
                    )
            else:
                Debug.error_exit(
                    "'origin_dataset_ver' must be one of 'AU.1mm/AU.3mm/MDA'!"
                )

        elif dataset_ver not in [DatasetVer.AU_1MM, DatasetVer.AU_3MM, DatasetVer.MDA]:
            Debug.error_exit("'dataset_ver' must be one of 'AU.1mm/AU.3mm/MDA'!")

        return dataset_ver

    def _is_valid_dataset_part(
        self,
        dataset_part: str,
        dataset_ver: str = None,
    ):
        if dataset_part not in [
            DatasetPart.TRAIN,
            DatasetPart.VALID,
            DatasetPart.TEST,
            DatasetPart.TEST_INTER,
            DatasetPart.TEST_EXTER,
        ]:
            Debug.error_exit(
                "'dataset_part' must be one of 'train/valid/test/test.inter/test.exter'!"
            )

        # check dataset section based on dataset version
        if dataset_ver is not None:
            dataset_ver = self._is_valid_dataset_version(dataset_ver=dataset_ver)

            if dataset_ver == DatasetVer.MDA:
                if (
                    dataset_part == DatasetPart.TEST_INTER
                    or dataset_part == DatasetPart.TEST_EXTER
                ):
                    Debug.error_exit(
                        "Use 'test' instead of 'test.inter/test.exter' for MDA dataset!"
                    )

            elif dataset_ver == DatasetVer.AU_3MM or dataset_ver == DatasetVer.AU_1MM:
                if dataset_part == DatasetPart.TEST:
                    Debug.error_exit(
                        "Use 'test.inter/test.exter' instead of 'test' for AU dataset!"
                    )
