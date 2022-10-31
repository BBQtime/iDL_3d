import os
import warnings
import shutil
import torch
import json
import random
import platform
import imageio
import hashlib
import unicodedata
import cc3d
import imgaug as ia
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from numpy import ndarray
from torch import Tensor
from PyQt5 import QtWidgets
from typing import Union
from natsort import natsorted


def exit_app(msg: str = ""):
    if msg == "" or msg is None:
        msg = "debug exit"
    else:
        msg = str(msg)
    print(msg)
    assert 0, msg


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
        )


# after json loaded, key(int) will become string
def load_json(path: str) -> dict:
    with open(path, mode="r") as json_file:
        data = json.load(json_file)
    # call "save_json" to sort data by key
    save_json(data=data, path=path)
    return data


# [1,2,3,4] -> "1,2,3,4"
def list_to_str(input_list: list, split_symbol: str = ",") -> str:
    split_symbol = str(split_symbol)
    return split_symbol.join(str(i) for i in input_list)


# "1,2,3,4" -> [1,2,3,4]
def str_to_list(input_str: str, split_symbol: str = ",") -> list:
    input_str = str(input_str)
    split_symbol = str(split_symbol)
    return input_str.split(",")


def get_dict_keys(input_dict: dict):
    return list(input_dict.keys())


# {"0": [a, b], "1": [c, d], "2": [e]} -> [a, b, c, d, e]
def dict_to_list(input_dict: dict):
    output_list = []
    for cur_key in input_dict:
        for cur_value in input_dict[cur_key]:
            output_list.append(cur_value)
    return output_list


def show_img(img: Union[ndarray, Tensor], print_info: bool = False):
    if print_info:
        print("image data type:", type(img))
        print("image shape:", img.shape)
        print("image max value:", img.max())
        print("image min value:", img.min())

    if isinstance(img, Tensor):
        # detach: return a tensor share the same memory but without grad
        img = img.detach().cpu().numpy()
    if len(img.shape) == 2:
        ia.imshow(img)
    elif len(img.shape) == 3:
        ia.imshow(img[img.shape[0] // 2])
    elif len(img.shape) == 4:
        ia.imshow(img[img.shape[0] // 2][img.shape[1] // 2])


def rename_file(base_path: str, old_name: str, new_name: str):
    old_path = os.path.join(base_path, old_name)
    new_path = os.path.join(base_path, new_name)
    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        return True
    else:
        return False


def delete_file(path: str):
    if os.path.exists(path):
        os.remove(path)
        return True
    else:
        return False


def create_folder(path: str, overwrite: bool = False):
    if overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
    return path


def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
        return True
    else:
        return False


def rename_folder(base_path: str, old_name: str, new_name: str) -> bool:
    rename_file(base_path, old_name, new_name)


def delete_folder(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    else:
        return False


def get_file_md5(file_path: str):
    with open(file_path, "rb") as fp:
        md5_obj = hashlib.md5()
        md5_obj.update(fp.read())
        file_md5 = md5_obj.hexdigest()
        # print(file_md5)
        return file_md5


def get_file_sha1(file_path: str):
    with open(file_path, "rb") as fp:
        sha1_obj = hashlib.sha1()
        sha1_obj.update(fp.read())
        file_sha1 = sha1_obj.hexdigest()
        # print(file_sha1)
        return file_sha1


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


def to_pct(input_num: float):
    input_num = float(input_num)
    output_str = keep_decimal(input_num=input_num * 100, keep_dec_num=2)
    output_str = str(output_str) + "%"
    return output_str


def shuffle_list(input_list: list, seed: int):
    # sort before shuffle, make sure to get same list using same seed
    input_list = natsorted(input_list)
    if seed is not None:
        random_state = random.getstate()
        random.seed(seed)
        random.shuffle(input_list)
        random.setstate(random_state)
    else:
        random.shuffle(input_list)
    return input_list


def __get_sub_items(
    folder_path: str,
    key_word: str = None,
    shuffle: bool = False,
    seed: int = None,
    select: str = "both",
):
    if not os.path.exists(folder_path):
        exit_app(
            'general:__get_sub_elems(): Input folder path "{}" does not exist'.format(
                folder_path
            )
        )
    sub_list = os.listdir(folder_path)
    # create del_list because it's not safe to
    # remove an element from current activated list in for loop
    if select != "both":
        sub_list_copy = sub_list.copy()
        for sub_name in sub_list:
            if select == "files":
                if not os.path.isfile(os.path.join(folder_path, sub_name)):
                    sub_list_copy.remove(sub_name)
            elif select == "folders":
                if os.path.isfile(os.path.join(folder_path, sub_name)):
                    sub_list_copy.remove(sub_name)
                if sub_name == ".ipynb_checkpoints":
                    sub_list_copy.remove(sub_name)
        sub_list = sub_list_copy.copy()

    if shuffle:
        # shuffle_list() includes natsorted()
        sub_list = shuffle_list(sub_list, seed)
    else:
        sub_list = natsorted(sub_list)

    # key word
    if key_word is not None:
        for i in sub_list.copy():
            if key_word not in i:
                sub_list.remove(i)

    return sub_list


def get_sub_items(
    folder_path: str, key_word: str = None, shuffle: bool = False, seed: int = None
):
    sub_list = __get_sub_items(
        folder_path=folder_path,
        key_word=key_word,
        shuffle=shuffle,
        seed=seed,
        select="both",
    )
    return sub_list


def get_sub_files(
    folder_path: str, key_word: str = None, shuffle: bool = False, seed: int = None
):
    sub_list = __get_sub_items(
        folder_path=folder_path,
        key_word=key_word,
        shuffle=shuffle,
        seed=seed,
        select="files",
    )
    return sub_list


def get_sub_folders(
    folder_path: str, key_word: str = None, shuffle: bool = False, seed: int = None
):
    sub_list = __get_sub_items(
        folder_path=folder_path,
        key_word=key_word,
        shuffle=shuffle,
        seed=seed,
        select="folders",
    )
    return sub_list


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return True
    else:
        return False


def check_limit(
    in_value: Union[float, int],
    low_limit: Union[float, int] = None,
    up_limit: Union[float, int] = None,
):
    if low_limit is not None:
        if in_value < low_limit:
            in_value = low_limit
    if up_limit is not None:
        if in_value > up_limit:
            in_value = up_limit
    return in_value


def get_avg_value(input_data: Union[list, dict]) -> float:
    if isinstance(input_data, list):
        return sum(input_data) / len(input_data)
    elif isinstance(input_data, dict):
        return sum(input_data.values()) / len(input_data)
    else:
        return input_data


# def cross_join_lists(list1, list2):
#     newlist = []
#     len1 = len(list1)
#     len2 = len(list2)
#     for i in range(max(len1, len2)):
#         if i < len1:
#             newlist.append(list1[i])
#         if i < len2:
#             newlist.append(list2[i])
#     return newlist


def print_line(len: int = 50):
    print("=" * len)


def load_nii(nii_path: str):
    img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(img)
    return img


def save_nii(np_data: ndarray, save_path: str, spacing: tuple = None):
    itk_img = sitk.GetImageFromArray(np_data)
    if spacing is None:
        itk_img.SetSpacing(NII_SPACING)
    else:
        itk_img.SetSpacing(spacing)
    sitk.WriteImage(itk_img, save_path)
    return save_path


def save_img(save_path: str, np_data: ndarray, extension_name: str = ".png"):
    if not save_path.endswith(extension_name):
        save_path += extension_name
    imageio.imwrite(save_path, np_data)
    return save_path


# def tensor_save_img(tensor_data, save_path):
#     vutils.save_image(
#         tensor_data,
#         save_path,
#         normalize=True,
#     )


# def clear_linux_trash():
#     if platform.system().lower() == "linux":
#         clear_folder("/home/alan/.local/share/Trash/files/")
#         clear_folder("/home/alan/.local/share/Trash/info/")


def get_combox_content(combox: QtWidgets.QComboBox):
    content_list = []
    for i in range(combox.count()):
        content_list.append(combox.itemText(i))
    return content_list


def binarize_img(
    img: Union[ndarray, Tensor], threshold: float = 0.5
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


def get_connected_components(img: ndarray) -> list:
    img = binarize_img(img)
    all_cc, num_cc = cc3d.connected_components(img, connectivity=18, return_N=True)
    output_cc_list = []
    for segid in range(1, num_cc + 1):
        cur_cc = all_cc * (all_cc == segid)
        # batch normalize
        cur_cc = cur_cc / segid
        # save_nii(cur_cc, "F:/cc_{}.nii".format(segid), NII_SPACING)
        output_cc_list.append(cur_cc)
    return output_cc_list


def used_gpu_count() -> int:
    if DEVICE == torch.device("cpu"):
        return 0
    else:
        return torch.cuda.device_count()


def is_number(i) -> bool:
    if i is None:
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


def change_char_in_str(input_str: str, idx: int, new_val: str):
    input_str = list(input_str)
    input_str[idx] = new_val
    return "".join(input_str)


def sort_dict_by_value(input_dict: dict):
    return {k: v for k, v in sorted(input_dict.items(), key=lambda item: item[1])}


def get_list_avg(input_list: list):
    return sum(input_list) / len(input_list)


PROJ_PATH = None
DEVICE = None
MAX_BATCH_SIZE = None
NUM_WORKERS = None
IMG_SIZE = None
NII_SPACING = None
CNN_STATE_DICT_ONLY = None
DATASET_FOLDER = None
DATASET_SPLITTING_JSON = None
BASELINE_HYPER_JSON = None
IDL_HYPER_JSON = None
TRAIN_RESULTS_FOLDER = None
BASELINE_TENSORBOARD_FOLDER = None
IDL_TENSORBOARD_FOLDER = None


def __general_init():
    global PROJ_PATH
    global DEVICE
    global MAX_BATCH_SIZE
    global NUM_WORKERS
    global IMG_SIZE
    global NII_SPACING
    global CNN_STATE_DICT_ONLY
    global DATASET_FOLDER
    global DATASET_SPLITTING_JSON
    global BASELINE_HYPER_JSON
    global IDL_HYPER_JSON
    global TRAIN_RESULTS_FOLDER
    global BASELINE_TENSORBOARD_FOLDER
    global IDL_TENSORBOARD_FOLDER

    PROJ_PATH = os.path.dirname(os.path.dirname(__file__))
    __json_data = load_json(os.path.join(PROJ_PATH, "settings.json"))

    # use CPU
    if __json_data["use.gpu"] is False:
        DEVICE = torch.device("cpu")
    # use GPU
    else:
        # choose GPU (must come first before any code related to cuda/gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = __json_data["cuda.visible.devices"]

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

    # Windows or Linux
    if platform.system().lower() == "windows":
        MAX_BATCH_SIZE = __json_data["max.batch.size.win"]  # g.IMG_SIZE 256
        NUM_WORKERS = __json_data["num.workers.win"]

    elif platform.system().lower() == "linux":
        MAX_BATCH_SIZE = __json_data["max.batch.size.linux"]  # g.IMG_SIZE 256
        NUM_WORKERS = __json_data["num.workers.linux"]

    # # Google Colab
    # if os.getcwd() == "/content":
    #     from google.colab import drive
    #     drive.mount("/content/drive")
    #     PROJ_PATH = "/content/drive/My Drive/alan/iDL/"
    #     MAX_BATCH_SIZE =  32  # IMG_SIZE 256

    # const values
    if used_gpu_count() > 1:
        MAX_BATCH_SIZE *= 2

    IMG_SIZE = []
    for i in str_to_list(__json_data["img.size"]):
        IMG_SIZE.append(int(i))
    IMG_SIZE = tuple(IMG_SIZE)

    # make sure all elements in NII_SPACING are numbers
    NII_SPACING = []
    for i in str_to_list(__json_data["nii.spacing"]):
        NII_SPACING.append(float(i))
    NII_SPACING = tuple(NII_SPACING)

    # Pytorch save/load entire cnn or weight only
    CNN_STATE_DICT_ONLY = __json_data["cnn.state.dict.only"]

    DATASET_FOLDER = __json_data["dataset.folder"]
    DATASET_SPLITTING_JSON = os.path.join(PROJ_PATH, __json_data["dataset.split.json"])
    BASELINE_HYPER_JSON = os.path.join(PROJ_PATH, __json_data["baseline.hyper.json"])
    IDL_HYPER_JSON = os.path.join(PROJ_PATH, __json_data["idl.hyper.json"])
    TRAIN_RESULTS_FOLDER = os.path.join(PROJ_PATH, __json_data["train.results.folder"])
    BASELINE_TENSORBOARD_FOLDER = os.path.join(
        PROJ_PATH, __json_data["baseline.tensorboard.folder"]
    )
    IDL_TENSORBOARD_FOLDER = os.path.join(
        PROJ_PATH, __json_data["idl.tensorboard.folder"]
    )


# general initialization
__general_init()
