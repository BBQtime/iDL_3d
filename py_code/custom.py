import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import string
import sys
import unicodedata
import warnings
from datetime import datetime
from typing import Union

import cc3d
import cv2
import numpy as np
import SimpleITK as sitk
import torch
from natsort import natsorted
from numpy import ndarray
from str_lib import DatasetVer
from torch import Tensor


# nested dictionary
# (1) new_dict=Dict(origin_dict), change new_dict will not change origin_dict
# (2) better make all keys "str", because Json.load() will change key type(int type) into string
class Dict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def copy(self):
        return copy.deepcopy(self)

    def keys(self):
        return List(super().keys())

    def key_with_max_value(self):
        return max(super().keys(), key=(lambda k: self[k]))

    def key_with_min_value(self):
        return min(super().keys(), key=(lambda k: self[k]))

    def sort_by_value(self, reverse: bool = False):
        sorted_dict = sorted(self.items(), key=lambda item: item[1], reverse=reverse)
        self.clear()
        self.update(sorted_dict)

    # {"0": [a, b], "1": [c, d], "2": [e]} -> [a, b, c, d, e]
    def to_list(self):
        output_list = List()
        for value in self.values():
            if isinstance(value, dict):
                sub_list = Dict(value).to_list()
                output_list.extend(sub_list)
            elif isinstance(value, list):
                output_list.extend(value)
            else:
                output_list.append(value)
        return output_list


class List(list):
    # slicing operations on List still return an instance of List
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return List(result)
        else:
            return result

    def __init__(self, *args):
        # normal init
        if len(args) == 0:
            super().__init__(*args)
        elif len(args) == 1:
            # str to list, split with comma: "1,2,3,4" -> ["1","2","3","4"]
            if isinstance(args[0], str):
                if args[0] == "":
                    super().__init__()
                else:
                    super().__init__(args[0].split(","))
            # init int and float like List(1) or List(2.0)
            elif isinstance(args[0], int) or isinstance(args[0], float):
                super().__init__([args[0]])
            else:
                super().__init__(*args)
        # normal init
        else:
            super().__init__(*args)

    # [1,2,3,4] -> "1,2,3,4"
    def to_str(self) -> str:
        return ",".join(str(i) for i in self)

    def copy(self):
        return copy.deepcopy(self)

    def find_identical_items(self, other_list: list):
        identical_items = set(self) & set(other_list)
        self[:] = List(identical_items)
        self.sort()

    def shuffle(self, seed: int = None):
        # sort before shuffle, ensure to get specific results using specific seed
        self.sort()
        if seed is not None:
            random_state = random.getstate()
            random.seed(seed)
            random.shuffle(self)
            random.setstate(random_state)
        else:
            random.shuffle(self)

    def sort(self, reverse: bool = False):
        super().__init__(natsorted(self, reverse=reverse))

    def remove_duplicates(self):
        self[:] = List(set(self))


class Value:
    EPS = sys.float_info.epsilon

    def random_str(length=10):
        # Generate a random string of fixed length
        letters = string.ascii_letters
        return "".join(random.choice(letters) for i in range(length))

    def replace_char(input_str: str, idx: int, new_char: str) -> str:
        return input_str[:idx] + new_char + input_str[idx + 1 :]

    def keep_decimal(input_num: Union[float, str], keep_dec_num: int = 0):
        output_num = str(input_num)
        keep_range = output_num.find(".")
        if keep_range > -1:
            keep_range = keep_range + keep_dec_num
            if keep_dec_num > 0:
                keep_range += 1
            output_num = output_num[0:keep_range]
        if isinstance(input_num, float):
            return float(output_num)
        else:
            return output_num

    def to_pct(input_num: Union[float, str]) -> str:
        input_num = float(input_num)
        output_str = Value.keep_decimal(input_num=input_num * 100, keep_dec_num=2)
        output_str = str(output_str) + "%"
        return output_str

    def limit_range(value, limit: tuple):
        low_limit = limit[0]
        up_limit = limit[1]
        if low_limit is not None:
            if value < low_limit:
                value = low_limit
        if up_limit is not None:
            if value > up_limit:
                value = up_limit
        return value

    def is_number(i) -> bool:
        if (
            i is None
            or isinstance(i, list)
            or isinstance(i, dict)
            or isinstance(i, str)
            or math.isnan(i)
        ):
            return False
        try:
            float(i)
            return True
        except ValueError:
            pass
        try:
            unicodedata.numeric(i)
            return True
        except (TypeError, ValueError):
            pass
        return False

    # origin data will not be replaced in this function
    def __is_valid_for_avg_median_calculation(origin_data: Union[list, dict, tuple]):
        if isinstance(origin_data, dict):
            data_list = Dict(origin_data).to_list()
        elif isinstance(origin_data, tuple):
            data_list = list(origin_data)
        elif isinstance(origin_data, list):
            data_list = origin_data.copy()
        else:
            Debug.error_exit("Input data should be a list, dict or tuple instance!")
        return data_list

    def median(origin_data: Union[list, dict, tuple]) -> float:
        data_list = Value.__is_valid_for_avg_median_calculation(origin_data)
        data_list = [i if i is not None else math.nan for i in data_list]
        if len(data_list) == 0:
            return None
        else:
            answer = statistics.median(data_list)
            if math.isnan(answer):
                return None
            else:
                return answer

    def avg(origin_data: Union[list, dict, tuple]) -> float:
        data_list = Value.__is_valid_for_avg_median_calculation(origin_data)
        # remove non-number in the list
        data_list = [i for i in data_list if Value.is_number(i)]
        if len(data_list) == 0:
            return None
        else:
            return statistics.mean(data_list)


class Img:
    def binarize(
        img: Union[ndarray, Tensor],
        threshold: float = 0.5,
    ) -> Union[ndarray, Tensor]:
        if isinstance(img, ndarray):
            ones = np.ones_like(img)
            zeros = np.zeros_like(img)
            img = np.where(img >= threshold, ones, zeros)
        elif isinstance(img, Tensor):
            ones = torch.ones_like(img)
            zeros = torch.zeros_like(img)
            img = torch.where(img >= threshold, ones, zeros)
        return img

    def normalize(img: ndarray) -> ndarray:
        # make min value=0
        img = img - img.min()
        # make range between [0-1]
        img /= img.max()
        return img

    def estimate_window_level(img: ndarray):
        # Ensure data is in float format for accurate calculations
        data_flat = img.ravel()

        # Exclude zero or negative values if present, which is common in PET or MR images
        data_flat = data_flat[data_flat > 0]

        # Calculate percentiles to exclude extreme values
        # You can adjust these percentiles as needed
        lower_percentile, upper_percentile = np.percentile(data_flat, [2, 99])

        window = upper_percentile - lower_percentile
        level = lower_percentile + window / 2

        return window, level

    def __img_windowing(img, window: int, level: int):
        high = level + window / 2
        low = level - window / 2
        img = np.where(img > high, high, img)
        img = np.where(img < low, low, img)
        return img

    # ct windowing (only focus on soft tissue)
    def ct_windowing(img):
        # in origin_dicom, air is -1024, soft tissue is 40
        # in our ct img, air is 0, soft tissue is 40+1024
        window = 350
        level = 40 + 1024
        img = Img.__img_windowing(img=img, window=window, level=level)
        return img

    def img_windowing(img):
        window, level = Img.estimate_window_level(img)
        img = Img.__img_windowing(img=img, window=window, level=level)
        return img

    # max size: 89 283 280
    def central_crop(img: ndarray, target_shape: tuple) -> ndarray:
        in_shape = Dict()
        in_shape["d"], in_shape["h"], in_shape["w"] = img.shape

        out_shape = Dict()
        out_shape["d"] = target_shape[0]
        out_shape["h"] = target_shape[1]
        out_shape["w"] = target_shape[2]

        if (
            in_shape["d"] > out_shape["d"]
            or in_shape["h"] > out_shape["h"]
            or in_shape["w"] > out_shape["w"]
        ):
            start_point = Dict()

            for i in ["w", "h", "d"]:
                if in_shape[i] > out_shape[i]:
                    # crop 1 more line on direction 1 (away from staring point)
                    start_point[i] = (in_shape[i] - out_shape[i]) // 2
                else:
                    start_point[i] = 0

            img = img[
                start_point["d"] : start_point["d"] + out_shape["d"],
                start_point["h"] : start_point["h"] + out_shape["h"],
                start_point["w"] : start_point["w"] + out_shape["w"],
            ]

        return img

    def central_pad(img: ndarray, target_shape: tuple) -> ndarray:
        in_shape = Dict()
        in_shape["d"], in_shape["h"], in_shape["w"] = img.shape

        out_shape = Dict()
        out_shape["d"] = target_shape[0]
        out_shape["h"] = target_shape[1]
        out_shape["w"] = target_shape[2]

        pad = Dict()
        for i in ["w", "h", "d"]:
            pad[i][0] = 0
            pad[i][1] = 0

        for i in ["w", "h", "d"]:
            if out_shape[i] > in_shape[i]:
                cur_pad = out_shape[i] - in_shape[i]
                pad[i][0] = int(cur_pad / 2)
                if cur_pad % 2 == 0:
                    pad[i][1] = pad[i][0]
                else:
                    # pad 1 more line on direction 1 (away from staring point)
                    pad[i][1] = pad[i][0] + 1

        img = np.pad(
            img,
            (
                (pad["d"][0], pad["d"][1]),
                (pad["h"][0], pad["h"][1]),
                (pad["w"][0], pad["w"][1]),
            ),
            "constant",
            constant_values=0,  # constant_values=0 means black padding
        )
        return img

    def central_pad_and_crop(img: ndarray, target_shape: tuple):
        img = Img.central_pad(img, target_shape)
        img = Img.central_crop(img, target_shape)
        return img

    def connected_components(img: ndarray) -> List:
        img = Img.binarize(img)
        all_cc, num_cc = cc3d.connected_components(img, connectivity=18, return_N=True)
        output_cc_list = List()
        for segid in range(1, num_cc + 1):
            cur_cc = all_cc * (all_cc == segid)
            # batch normalize
            cur_cc = cur_cc / segid
            output_cc_list.append(cur_cc)
        return output_cc_list

    def find_random_point(binary_img) -> list:
        binary_img = Img.binarize(binary_img)

        # pos of all nonzero voxels in the img
        # shape of pos:[nonzero_count, img_dim]
        non_zero_pos = np.argwhere(binary_img)

        # if no nonzero elements, return None
        if non_zero_pos.size == 0:
            return None

        # select a random coordinate from the list of nonzero elements
        random_idx = random.randint(0, non_zero_pos.shape[0] - 1)
        random_pos = non_zero_pos[random_idx]

        # Return the tuple (x, y, z) corresponding to the random coordinate
        return random_pos

    def gray_to_rgb(gray_img: ndarray) -> ndarray:
        # rgb_img = np.uint8((gray_img - gray_img.min()) / gray_img.ptp() * 255.0)
        gray_img = cv2.convertScaleAbs(gray_img, alpha=255.0)
        # after cv2.cvtColor, rgb_img has 3 channels, but still a ndarray
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        return rgb_img

    def gray_to_colormap(gray_img: ndarray) -> ndarray:
        # Scale to [0, 255] and convert to uint8
        scaled_img = np.clip(gray_img * 255, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(
            scaled_img,
            cv2.COLORMAP_HOT,  # black to blue to white
        )
        return color_map

    # use this in case there is no gtvn or gtvs nii file
    def load_labels(
        dataset_dir: str,
        patient: str,
        nii_load_func=None,  # ui will use this param, with its own nii load function
    ):
        if nii_load_func is None:
            nii_load_func = Nii.load

        paths = Dict()
        for i in ["s", "t", "n"]:
            paths["gtv{}".format(i)] = os.path.join(
                dataset_dir, "HNCDL_{}_GTV{}.nii".format(patient, i)
            )
        labels = Dict()

        # load gtvt
        labels["gtvt"] = nii_load_func(paths["gtvt"], binary=True)

        # load gtvn
        if os.path.exists(paths["gtvn"]):
            labels["gtvn"] = nii_load_func(paths["gtvn"], binary=True)
        else:
            labels["gtvn"] = np.zeros_like(labels["gtvt"])

        # load gtvs
        if os.path.exists(paths["gtvs"]):
            labels["gtvs"] = nii_load_func(paths["gtvs"], binary=True)
        else:
            labels["gtvs"] = np.maximum(labels["gtvt"], labels["gtvn"])

        return labels


class Nii:
    def load(
        path: str,
        binary: bool = False,
        dim: int = 3,
        return_info: bool = False,
    ) -> ndarray:
        img = sitk.ReadImage(path)
        if return_info:
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
        img = sitk.GetArrayFromImage(img)
        img = img.astype(np.float32)
        if binary:
            img = Img.binarize(img)
        if dim > 0 and len(img.shape) > dim:
            for i in range(len(img.shape) - dim):
                img = np.squeeze(img, axis=0)
        if return_info:
            return img, spacing, origin
        else:
            return img

    def save(
        img: Union[ndarray, Tensor],
        save_path: str,
        spacing: tuple = None,
        origin: tuple = None,
    ):
        # tensor to ndarray
        if isinstance(img, Tensor):
            # detach: return a tensor share the same memory but without grad
            img = img.detach().cpu().numpy()

        # squeeze to 3d img
        if len(img.shape) > 3:
            for i in range(len(img.shape) - 3):
                if img.shape[i] == 1:
                    img = np.squeeze(img, axis=0)
                else:
                    img = img[0]

        itk_img = sitk.GetImageFromArray(img)

        # if copy_info_from is not None:
        #     itk_img.CopyInformation(sitk.ReadImage(copy_info_from))

        if spacing is not None:
            itk_img.SetSpacing(spacing)
        if origin is not None:
            itk_img.SetOrigin(origin)

        sitk.WriteImage(itk_img, save_path)
        return save_path


class Json:
    def save(data: dict, path: str):
        with open(path, mode="w", encoding="utf-8") as json_file:
            # ensure_ascii == false, non-ASCII characters is available
            # skipkeys=True keys are not str will be skipped
            json.dump(
                data,
                json_file,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
                skipkeys=False,
            )

    # after json loaded, key(int) will become string
    def load(path: str) -> Dict:
        with open(path, mode="r") as json_file:
            data = json.load(json_file)
        data = Dict(data)
        # call "Json.save" to sort key
        Json.save(data=data, path=path)
        return data


class Dir:
    def create(path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def clear(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        else:
            Debug.error_exit("'path' is not a dir!")

    def rename(parent_dir: str, old_name: str, new_name: str):
        old_path = os.path.join(parent_dir, old_name)
        new_path = os.path.join(parent_dir, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)

    def get_file_md5(file_path: str):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as fp:
                md5_obj = hashlib.md5()
                md5_obj.update(fp.read())
                file_md5 = md5_obj.hexdigest()
                # print(file_md5)
                return file_md5
        else:
            Debug.error_exit("'file_path' is not a file!")

    def get_file_sha1(file_path: str):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as fp:
                sha1_obj = hashlib.sha1()
                sha1_obj.update(fp.read())
                file_sha1 = sha1_obj.hexdigest()
                # print(file_sha1)
                return file_sha1
        else:
            Debug.error_exit("'file_path' is not a file!")

    def delete(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    def __get_sub_items(
        input_dir: str,
        full_path: bool,
        key_word: str,
        shuffle: bool,
        seed: int,
        select: str,
    ) -> List:
        sub_list = List(os.listdir(input_dir))

        if select != "both":
            for sub_name in sub_list.copy():
                if select == "files":
                    if not os.path.isfile(os.path.join(input_dir, sub_name)):
                        sub_list.remove(sub_name)
                elif select == "folders":
                    if os.path.isfile(os.path.join(input_dir, sub_name)):
                        sub_list.remove(sub_name)

        if shuffle:
            sub_list.shuffle(seed)
        else:
            sub_list.sort()

        if key_word != "":
            for i in sub_list.copy():
                if key_word not in i:
                    sub_list.remove(i)

        if full_path:
            for i in range(len(sub_list)):
                sub_list[i] = os.path.join(input_dir, sub_list[i])

        return sub_list

    def get_sub_items(
        input_dir: str,
        key_word: str = "",
        full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ) -> List:
        sub_list = Dir.__get_sub_items(
            input_dir=input_dir,
            full_path=full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="both",
        )
        return sub_list

    def get_sub_files(
        input_dir: str,
        key_word: str = "",
        full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ) -> List:
        sub_list = Dir.__get_sub_items(
            input_dir=input_dir,
            full_path=full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="files",
        )
        return sub_list

    def get_sub_dirs(
        input_dir: str,
        key_word: str = "",
        full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ) -> List:
        sub_list = Dir.__get_sub_items(
            input_dir=input_dir,
            full_path=full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="folders",
        )
        return sub_list

    def __walk_sub_dirs(input_dir: str) -> List:
        sub_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
        for sub_dir in sub_dirs.copy():
            sub_dirs.extend(Dir.__walk_sub_dirs(sub_dir))
        return sub_dirs

    def walk_sub_dirs(input_dir: str, key_word: str = "", suffle=False) -> List:
        sub_dirs = List()
        for sub_dir in Dir.__walk_sub_dirs(input_dir):
            if key_word == "" or key_word in sub_dir:
                sub_dirs.append(sub_dir)
        if suffle:
            sub_dirs.shuffle()
        else:
            sub_dirs.sort()
        return sub_dirs


class GPU:
    def clear_cache():
        # garbage collector
        gc.collect()
        # empty torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def used_count() -> int:
        if Global.DEVICE == torch.device("cpu"):
            return 0
        else:
            return torch.cuda.device_count()


class Debug:
    DELETE_FLAG = "delete.flag"

    def error_exit(err_msg: str = ""):
        assert 0, err_msg

    def clear_debug_data():
        for i in Dir.walk_sub_dirs(
            Global.TRAIN_RESULTS_DIR, key_word=Debug.DELETE_FLAG
        ):
            Dir.delete(i)
        Dir.clear(os.path.join(Global.PROJ_DIR, "debug"))

    def clear_linux_trash():
        if Global.is_linux():
            Dir.clear("/home/alan/.local/share/Trash/files/")
            Dir.clear("/home/alan/.local/share/Trash/info/")


class Timer:
    def cur_time_str(hide_microsecond=True) -> str:
        cur_time = datetime.now()
        if hide_microsecond:
            cur_time = cur_time.replace(microsecond=0)
        cur_time = str(cur_time)
        cur_time = cur_time.replace(":", ".")
        cur_time = cur_time.replace("-", ".")
        cur_time = cur_time.replace(" ", ".")
        return cur_time


class Global:
    def combine_pred_correction(origin_pred, correction, correction_mask):
        if correction is None or correction_mask is None:
            if origin_pred is None:
                return None
            else:
                return origin_pred.copy()
        else:
            if origin_pred is None:
                return correction * correction_mask
            else:
                return np.maximum(
                    origin_pred * (1 - correction_mask),
                    correction * correction_mask,
                )

    def is_linux():
        return platform.system().lower() == "linux"

    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    DEBUG_DIR = os.path.join(PROJ_DIR, "debug")

    __settings = Json.load(os.path.join(PROJ_DIR, "settings_global.json"))

    # use CPU
    if __settings["cuda.visible.devices"] == "":
        DEVICE = torch.device("cpu")
    # use GPU
    else:
        # choose GPU (must come first before any code related to cuda/gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = __settings["cuda.visible.devices"]

        # set main device cuda:0, for multiple GPU to avoid following error:
        # RuntimeError: module must have its parameters and buffers on
        # device cuda:0 (device_ids[0]) but found one of them on device: cuda:1
        if torch.cuda.device_count() > 1:
            DEVICE = torch.device("cuda:0")

        elif torch.cuda.device_count() == 1:
            DEVICE = torch.device("cuda")

        else:  # torch.cuda.device_count() < 1:
            DEVICE = torch.device("cpu")

    # hide warning
    warnings.filterwarnings("ignore")

    DATASET_DIR = Dict()

    for i in [
        DatasetVer.AU,
        DatasetVer.OBS_STUDY,
        DatasetVer.MDA,
    ]:
        platform_name = "linux" if is_linux() else "windows"
        DATASET_DIR[i] = __settings["dataset.dir.{}.{}".format(platform_name, i)]

    # window doesn't support pytorch multi-thread
    NUM_WORKERS = __settings["num.workers"] if is_linux() else 0

    # IMG_SHAPE (Depth, Height, Width)
    IMG_SHAPE = List(__settings["img.shape"])
    IMG_SHAPE = tuple(int(k) for k in IMG_SHAPE)

    # NII_SPACING (Width, Height, Depth)
    NII_SPACING = List(__settings["nii.spacing"])
    NII_SPACING = tuple(float(k) for k in NII_SPACING)

    # dataset splitting
    DATASET_SPLIT_JSON_PATH = Dict()
    for i in [
        DatasetVer.AU,
        DatasetVer.OBS_STUDY,
        DatasetVer.MDA,
    ]:
        DATASET_SPLIT_JSON_PATH[i] = os.path.join(
            PROJ_DIR, __settings["dataset.split.json.{}".format(i)]
        )

    HYPER_JSON_PATH = Dict()
    for i in ["baseline", "idl.gtvt", "idl.gtvn"]:
        HYPER_JSON_PATH[i] = os.path.join(
            PROJ_DIR, __settings["hyper.json.{}".format(i)]
        )

    DATASET_FOLDS = __settings["dataset.folds"]
    TRAIN_RESULTS_DIR = os.path.join(PROJ_DIR, __settings["train.results.dir"])

    if is_linux():
        FONT_STYLE = "font-size: {}pt;".format(10)
    else:
        FONT_STYLE = "font-size: {}pt;".format(11)
    FONT_STYLE = "font-family: Arial;" + FONT_STYLE
    FONT_STYLE += "font-weight: bold;color: white;"

    TEXT_HEIGHT = 20 if is_linux() else 30
    SLIDER_HEIGHT = 16 if is_linux() else 21
