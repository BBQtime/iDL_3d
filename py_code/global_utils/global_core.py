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
import time
import unicodedata
import warnings
from datetime import datetime
from multiprocessing import Lock
from pathlib import Path
from typing import Union

import cc3d
import cv2
import numpy as np
import SimpleITK as sitk
import torch
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetVer, ErrMsg, MdaObs
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm

# Create a multiprocessing lock (shared across processes)
file_access_lock = Lock()


def error_exit(err_msg: str = ""):
    assert 0, err_msg


def is_linux():
    return platform.system().lower() == "linux"


# Generate a random string of specified length
def generate_random_str(len=10):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(len))


def replace_char_in_str(input_str: str, idx: int, new_char: str) -> str:
    return input_str[:idx] + new_char + input_str[idx + 1 :]


def round_decimal(input_num: Union[float, str], decimal_count: int = 0):
    output_num = str(input_num)
    keep_range = output_num.find(".")
    if keep_range > -1:
        keep_range = keep_range + decimal_count
        if decimal_count > 0:
            keep_range += 1
        output_num = output_num[0:keep_range]
    if isinstance(input_num, float):
        return float(output_num)
    else:
        return output_num


def format_as_pct(input_num: Union[float, str]) -> str:
    input_num = float(input_num)
    output_str = round_decimal(input_num=input_num * 100, decimal_count=2)
    output_str = str(output_str) + "%"
    return output_str


def clamp_value(value, limit: tuple):
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
        error_exit("Input data should be a list, dict or tuple instance!")
    return data_list


def calculate_median(origin_data: Union[list, dict, tuple]) -> float:
    data_list = __is_valid_for_avg_median_calculation(origin_data)
    # remove non-number in the list
    data_list = [i for i in data_list if is_number(i)]
    if len(data_list) == 0:
        return None
    else:
        return statistics.median(data_list)


def calculate_avg(origin_data: Union[list, dict, tuple]) -> float:
    data_list = __is_valid_for_avg_median_calculation(origin_data)
    # remove non-number in the list
    data_list = [i for i in data_list if is_number(i)]
    if len(data_list) == 0:
        return None
    else:
        return statistics.mean(data_list)


def binarize_img(
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


def normalize_img(img: ndarray) -> ndarray:
    # make min value=0
    img = img - img.min()
    # make range between [0-1]
    img /= img.max()
    return img


def __get_img_window_and_level(img: ndarray):
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


def __windowing_img(img, window: int, level: int):
    high = level + window / 2
    low = level - window / 2
    img = np.where(img > high, high, img)
    img = np.where(img < low, low, img)
    return img


# ct windowing (only focus on soft tissue)
def windowing_ct(ct_img):
    # in origin_dicom, air is -1024, soft tissue is 40
    # in our ct img, air is 0, soft tissue is 40+1024
    window = 350
    level = 40 + 1024
    ct_img = __windowing_img(img=ct_img, window=window, level=level)
    return ct_img


def windowing_img(img):
    window, level = __get_img_window_and_level(img)
    img = __windowing_img(img=img, window=window, level=level)
    return img


def center_crop_img(img: ndarray, target_shape: tuple) -> ndarray:
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


def center_pad_img(img: ndarray, target_shape: tuple) -> ndarray:
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


def center_align_img(img: ndarray, target_shape: tuple):
    img = center_pad_img(img, target_shape)
    img = center_crop_img(img, target_shape)
    return img


def get_connected_components(img: ndarray) -> List:
    img = binarize_img(img)
    all_cc, num_cc = cc3d.connected_components(img, connectivity=18, return_N=True)
    output_cc_list = List()
    for segid in range(1, num_cc + 1):
        cur_cc = all_cc * (all_cc == segid)
        # batch normalize
        cur_cc = cur_cc / segid
        output_cc_list.append(cur_cc)
    return output_cc_list


def get_random_nonzero_pos(binary_img) -> list:
    binary_img = binarize_img(binary_img)

    # pos of all nonzero voxels in the img
    # shape of pos:[nonzero_count, img_dim]
    nonzero_pos = np.argwhere(binary_img)

    # if no nonzero elements, return None
    if nonzero_pos.size == 0:
        return None

    # select a random coordinate from the list of nonzero elements
    random_idx = random.randint(0, nonzero_pos.shape[0] - 1)
    random_pos = nonzero_pos[random_idx]

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


def load_nii(
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
        img = binarize_img(img)
    if dim > 0 and len(img.shape) > dim:
        for i in range(len(img.shape) - dim):
            img = np.squeeze(img, axis=0)
    if return_info:
        return img, spacing, origin
    else:
        return img


def save_nii(
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


# update mda dataset structure
# from patient_1/label_1,2,3
# to patient_1_label_1, patient_1_label_2, patient_1_label_3
def mda_dataset_preprocess():
    old_dataset_dir = DATASET_DIR[DatasetVer.MDA]
    old_dataset_dir = Path(old_dataset_dir).parent
    old_dataset_dir = os.path.join(old_dataset_dir, "MDA_dataset_origin")

    new_dataset_dir = DATASET_DIR[DatasetVer.MDA]
    create_dir(new_dataset_dir)

    mda_obs_list = [MdaObs.AAA, MdaObs.DMEl, MdaObs.MRA, MdaObs.SA, MdaObs.YK]

    old_dataset_split = load_json(DATASET_SPLIT_PATH[DatasetVer.MDA])
    new_dataset_split = Dict()

    for i in old_dataset_split.keys():
        new_dataset_split[i] = List()

        for old_patient_name in tqdm(List(old_dataset_split[i])):
            old_patient_dir = os.path.join(old_dataset_dir, old_patient_name)

            for mda_obs in mda_obs_list:

                old_ct_path = os.path.join(
                    old_patient_dir,
                    "{}_CT.nii".format(old_patient_name),
                )
                old_mr1_path = os.path.join(
                    old_patient_dir,
                    "{}_T1dr.nii".format(old_patient_name),
                )
                old_mr2_path = os.path.join(
                    old_patient_dir,
                    "{}_T2dr.nii".format(old_patient_name),
                )

                old_gtvt_path = os.path.join(
                    old_patient_dir,
                    "{}_MR_GTV_P_BL_{}dr.nii".format(old_patient_name, mda_obs),
                )

                old_gtvn_path = os.path.join(
                    old_patient_dir,
                    "{}_MR_GTV_N_ALL_BL_{}dr.nii".format(old_patient_name, mda_obs),
                )

                # current observer has neither gtvt nor gtvn
                if not os.path.exists(old_gtvt_path) and not os.path.exists(
                    old_gtvn_path
                ):
                    continue

                else:
                    new_patient_name = old_patient_name + "_" + mda_obs

                    # (1)add into split json
                    new_dataset_split[i].append(new_patient_name)

                    # (2)copy files
                    new_patient_dir = os.path.join(new_dataset_dir, new_patient_name)
                    create_dir(new_patient_dir)

                    if os.path.exists(old_gtvt_path):
                        shutil.copy(
                            old_gtvt_path,
                            os.path.join(new_patient_dir, "GTVt.nii"),
                        )
                    if os.path.exists(old_gtvn_path):
                        shutil.copy(
                            old_gtvn_path,
                            os.path.join(new_patient_dir, "GTVn.nii"),
                        )

                    shutil.copy(
                        old_ct_path,
                        os.path.join(new_patient_dir, "CT.nii"),
                    )
                    shutil.copy(
                        old_mr1_path,
                        os.path.join(new_patient_dir, "T1dr.nii"),
                    )
                    shutil.copy(
                        old_mr2_path,
                        os.path.join(new_patient_dir, "T2dr.nii"),
                    )

        # list to str
        new_dataset_split[i] = new_dataset_split[i].to_str()

    save_json(new_dataset_split, DATASET_SPLIT_PATH[DatasetVer.MDA])


# use this function in case there is no gtvn or gtvs nii file
def load_gtv_labels(
    dataset_ver: str,
    patient: str,
    nii_load_func=None,  # ui will use this param, with its own nii load function
):
    dataset_dir = DATASET_DIR[dataset_ver]

    if nii_load_func is None:
        nii_load_func = load_nii

    paths = Dict()
    labels = Dict()

    if dataset_ver in [
        DatasetVer.AU,
        DatasetVer.AU_EXT,
        DatasetVer.OBS_STUDY,
        DatasetVer.NKI,
        DatasetVer.HECKTOR,
    ]:
        # load path
        for i in ["s", "t", "n"]:
            if dataset_ver in [DatasetVer.AU, DatasetVer.OBS_STUDY]:
                paths["gtv{}".format(i)] = os.path.join(
                    dataset_dir, "HNCDL_{}_GTV{}.nii".format(patient, i)
                )
            elif dataset_ver == DatasetVer.AU_EXT:
                paths["gtv{}".format(i)] = os.path.join(
                    dataset_dir,
                    "HNCDL_{}".format(patient),
                    "HNCDL_{}_GTV{}.nii".format(patient, i),
                )
            elif dataset_ver == DatasetVer.NKI:
                paths["gtv{}".format(i)] = os.path.join(
                    dataset_dir, patient, "{}_GTV{}.nii".format(patient, i)
                )
            elif dataset_ver == DatasetVer.HECKTOR:
                paths["gtv{}".format(i)] = os.path.join(
                    dataset_dir, "{}_GTV{}.nii.gz".format(patient, i)
                )
            else:
                error_exit(ErrMsg.DATASET_VER_INVALID)

        # load gtvt (for AU and OBS_STUDY dataset, there is always gtvt label)
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

    elif dataset_ver == DatasetVer.MDA:
        paths["gtvt"] = os.path.join(dataset_dir, patient, "GTVt.nii")
        paths["gtvn"] = os.path.join(dataset_dir, patient, "GTVn.nii")

        # current observer has neight gtvt nor gtvn
        if not os.path.exists(paths["gtvt"]) and not os.path.exists(paths["gtvn"]):
            error_exit("current MDA patient has neither GTVt nor GTVn!")

        # find both gtvt and gtvn, or one of them
        else:
            # (1) load existing label
            for i in ["gtvt", "gtvn"]:
                if os.path.exists(paths[i]):
                    labels[i] = nii_load_func(paths[i], binary=True)

            # (2) load non-existing label
            if not os.path.exists(paths["gtvt"]):
                labels["gtvt"] = np.zeros_like(labels["gtvn"])
            if not os.path.exists(paths["gtvn"]):
                labels["gtvn"] = np.zeros_like(labels["gtvt"])

            labels["gtvs"] = np.maximum(labels["gtvt"], labels["gtvn"])

            return labels

    else:
        error_exit(ErrMsg.DATASET_VER_INVALID)


# Custom encoder for numpy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def sort_json_dict(data: dict) -> Dict:
    # Convert the dictionary to a JSON string with sorted keys
    sorted_json_str = json.dumps(data, ensure_ascii=False, sort_keys=True)

    # Load it back into a dictionary
    sorted_dict = json.loads(sorted_json_str)

    # Convert it to your custom Dict type if needed
    return Dict(sorted_dict)


def save_json(data: dict, path: str):
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
            cls=NumpyEncoder,
        )


def load_setting_global_json(
    path: str, retries: int = 5, wait_time: float = 0.1
) -> Dict:
    # Retry mechanism in case of an empty or corrupt file
    for attempt in range(retries):
        with file_access_lock:  # Lock access to the file
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON file not found: {path}")

            with open(path, mode="r") as json_file:
                try:
                    data = json.load(json_file)
                    # Check if the data is valid and a non-empty dictionary
                    if isinstance(data, dict) and data:
                        break
                except json.JSONDecodeError as e:
                    # Handle decoding errors
                    if attempt == retries - 1:
                        raise ValueError(f"Error decoding JSON file: {path} - {str(e)}")
                    # else:
                    #     print(f"Attempt {attempt + 1}/{retries}: Error decoding JSON file. Retrying...")

        # Wait before retrying
        time.sleep(wait_time)
    else:
        # If after retries, the data is still invalid, raise an error
        raise ValueError(
            f"Failed to load valid JSON data from {path} after {retries} retries."
        )

    # Convert the loaded data to a custom Dict object
    data = Dict(data)

    # Call "save_json" to sort keys and potentially fix formatting issues
    data = sort_json_dict(data)
    # save_json(data=data, path=path)

    # Return the settings data as a Dict object
    return data


# after json loaded, key(int) will become string
def load_json(path: str, overwrite_sorted: bool = False) -> Dict:
    with open(path, mode="r") as json_file:
        data = json.load(json_file)
    data = Dict(data)
    # sort by keys
    data = sort_json_dict(data)
    # overwrite json with sorted keys
    if overwrite_sorted:
        save_json(data=data, path=path)
    return data


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clear_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    else:
        error_exit("'path' is not a dir!")


def rename_path(parent_dir: str, old_name: str, new_name: str):
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
        error_exit("'file_path' is not a file!")


def get_file_sha1(file_path: str):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            sha1_obj = hashlib.sha1()
            sha1_obj.update(fp.read())
            file_sha1 = sha1_obj.hexdigest()
            # print(file_sha1)
            return file_sha1
    else:
        error_exit("'file_path' is not a file!")


def delete_path(path: str):
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
    target: str,
) -> List:
    sub_list = List(os.listdir(input_dir))

    if target != "both":
        for sub_name in sub_list.copy():
            if target == "file":
                if not os.path.isfile(os.path.join(input_dir, sub_name)):
                    sub_list.remove(sub_name)
            elif target == "dir":
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
    sub_list = __get_sub_items(
        input_dir=input_dir,
        full_path=full_path,
        key_word=key_word,
        shuffle=shuffle,
        seed=seed,
        target="both",
    )
    return sub_list


def get_sub_files(
    input_dir: str,
    key_word: str = "",
    full_path: bool = False,
    shuffle: bool = False,
    seed: int = None,
) -> List:
    sub_list = __get_sub_items(
        input_dir=input_dir,
        full_path=full_path,
        key_word=key_word,
        shuffle=shuffle,
        seed=seed,
        target="file",
    )
    return sub_list


def get_sub_dirs(
    input_dir: str,
    key_word: str = "",
    full_path: bool = False,
    shuffle: bool = False,
    seed: int = None,
) -> List:
    sub_list = __get_sub_items(
        input_dir=input_dir,
        full_path=full_path,
        key_word=key_word,
        shuffle=shuffle,
        seed=seed,
        target="dir",
    )
    return sub_list


def __get_deep_dirs(input_dir: str) -> List:
    deep_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    for deep_dir in deep_dirs.copy():
        deep_dirs.extend(__get_deep_dirs(deep_dir))
    return deep_dirs


def get_deep_dirs(input_dir: str, key_word: str = "", suffle=False) -> List:
    deep_dirs = List()
    for deep_dir in __get_deep_dirs(input_dir):
        if key_word == "" or key_word in deep_dir:
            deep_dirs.append(deep_dir)
    if suffle:
        deep_dirs.shuffle()
    else:
        deep_dirs.sort()
    return deep_dirs


def delete_deep_files(file_name: str):
    deep_dirs = get_deep_dirs(PROJ_DIR)
    for deep_dir in deep_dirs:
        file_path = os.path.join(deep_dir, file_name)
        delete_path(file_path)


def clear_gpu_cache():
    # garbage collector
    gc.collect()
    # empty torch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def used_gpu_count(device_id: int) -> int:
    # (1) use CPU
    if __DEVICES[0] == torch.device("cpu"):
        return 0
    # (2) use multiple GPUs
    elif device_id <= -1 and len(__DEVICES) >= 2:
        return len(__DEVICES)
    # (3) use only 1 GPU, or only 1 visible GPU
    else:
        return 1


def get_device(device_id: int) -> torch.device:
    # (1) use CPU
    if __DEVICES[0] == torch.device("cpu"):
        return __DEVICES[0]
    # (2) more than 1 visible GPU devices
    elif len(__DEVICES) >= 2:
        # use all GPUs
        if device_id == -1:
            return __DEVICES[0]
        # only use 1 GPU
        else:
            return __DEVICES[device_id]
    # (3) only 1 visible GPU device
    elif len(__DEVICES) == 1:
        # Use "cuda:0" as the primary device, regardless the physical GPU ID
        return __DEVICES[0]

    # should never reach here
    else:
        error_exit("device setting error!")


def clear_debug_data():
    for i in get_deep_dirs(TRAIN_RESULTS_DIR, key_word=DELETE_FLAG):
        delete_path(i)
    clear_dir(os.path.join(PROJ_DIR, "debug"))


def clear_linux_trash():
    if is_linux():
        clear_dir("/home/alan/.local/share/Trash/files/")
        clear_dir("/home/alan/.local/share/Trash/info/")


def get_cur_time_str(hide_microsecond=True) -> str:
    cur_time = datetime.now()
    if hide_microsecond:
        cur_time = cur_time.replace(microsecond=0)
    cur_time = str(cur_time)
    cur_time = cur_time.replace(":", ".")
    cur_time = cur_time.replace("-", ".")
    cur_time = cur_time.replace(" ", ".")
    return cur_time


def find_contours(image_array):
    # Convert the numpy array to a SimpleITK Image
    image = sitk.GetImageFromArray(image_array)

    # Apply a thresholding filter to segment the image
    # Adjust the lower and upper thresholds according to your image's characteristics
    segmented = sitk.BinaryThreshold(
        image, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0
    )

    # Generate the contour for the segmented image
    contour_filter = sitk.BinaryContourImageFilter()
    contour_filter.SetFullyConnected(True)
    contour_filter.SetBackgroundValue(0)
    contour_image = contour_filter.Execute(segmented)

    # Convert back to numpy array
    contour_array = sitk.GetArrayFromImage(contour_image)

    return contour_array


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


def random_color():
    # random values for R, G, B
    return [random.random() for _ in range(3)]


PROJ_DIR = Path(os.path.dirname(os.path.dirname(__file__))).parent
DEBUG_DIR = os.path.join(PROJ_DIR, "debug")

__settings = load_setting_global_json(os.path.join(PROJ_DIR, "settings", "core.json"))

# devices setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# use GPU
if __settings["cuda.visible.devices"] in ["0", "1", "0,1"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = __settings["cuda.visible.devices"]
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        __DEVICES = [torch.device("cuda:{}".format(i)) for i in range(gpu_count)]
    else:  # Fallback to CPU if no GPU is found
        __DEVICES = [torch.device("cpu")]
# use CPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    __DEVICES = [torch.device("cpu")]


# hide warning
warnings.filterwarnings("ignore")

DATASET_DIR = Dict()
for __i in [
    DatasetVer.AU,
    DatasetVer.AU_EXT,
    DatasetVer.OBS_STUDY,
    DatasetVer.MDA,
    DatasetVer.NKI,
    DatasetVer.HECKTOR,
]:
    DATASET_DIR[__i] = __settings["dataset.dir"]["{}".format(__i)][
        "{}".format("linux" if is_linux() else "windows")
    ]

# window doesn't support pytorch multi-thread
if is_linux():
    NUM_WORKERS = __settings["num.workers"]["linux"]
else:
    NUM_WORKERS = __settings["num.workers"]["windows"]

# IMG_SHAPE (Depth, Height, Width)
IMG_SHAPE = List(__settings["img.shape"])
IMG_SHAPE = tuple(int(k) for k in IMG_SHAPE)

# NII_SPACING (Width, Height, Depth)
NII_SPACING = List(__settings["nii.spacing"])
NII_SPACING = tuple(float(k) for k in NII_SPACING)

# dataset splitting
DATASET_SPLIT_PATH = Dict()
for __i in [
    DatasetVer.AU,
    DatasetVer.AU_EXT,
    DatasetVer.OBS_STUDY,
    DatasetVer.MDA,
    DatasetVer.NKI,
    DatasetVer.HECKTOR,
]:
    DATASET_SPLIT_PATH[__i] = os.path.join(
        PROJ_DIR,
        "dataset_split",
        __settings["dataset.split.json"]["{}".format(__i)],
    )

HYPER_PATH = Dict()
for __i in ["baseline", "idl.gtvt", "idl.gtvn"]:
    HYPER_PATH[__i] = os.path.join(
        PROJ_DIR,
        "hyper",
        __settings["hyper.json"]["{}".format(__i)],
    )


DATASET_FOLDS = Dict()
for __i in [
    DatasetVer.AU,
    DatasetVer.MDA,
    DatasetVer.NKI,
    DatasetVer.HECKTOR,
]:
    DATASET_FOLDS[__i] = __settings["dataset.folds"]["{}".format(__i)]


TRAIN_RESULTS_DIR = os.path.join(PROJ_DIR, __settings["train.results.dir"])


if is_linux():
    FONT_STYLE = "font-size: {}pt;".format(10)
else:
    FONT_STYLE = "font-size: {}pt;".format(11)
FONT_STYLE = "font-family: Arial;" + FONT_STYLE
FONT_STYLE += "font-weight: bold;color: white;"

TEXT_HEIGHT = 20 if is_linux() else 30
SLIDER_HEIGHT = 16 if is_linux() else 21

DELETE_FLAG = "delete.flag"

EPS = sys.float_info.epsilon
