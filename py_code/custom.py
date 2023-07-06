import os
import sys
import math
import warnings
import statistics
import shutil
import torch
import json
import copy
import random
import platform
import hashlib
import unicodedata
import cc3d
import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from torch import Tensor
from typing import Union
from natsort import natsorted


# nested dictionary
# (1) new_dict=Dict(origin_dict), change new_dict will not change my_dict
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
        # str to list, split with comma: "1,2,3,4" -> ["1","2","3","4"]
        if len(args) == 1 and isinstance(args[0], str):
            if args[0] == "":
                super().__init__()
            else:
                super().__init__(args[0].split(","))
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


class ValueUtils:
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
        output_str = ValueUtils.keep_decimal(input_num=input_num * 100, keep_dec_num=2)
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
        if i is None or math.isnan(i):
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

    def median(data):
        return statistics.median(data)

    def avg(data: Union[list, dict, tuple]) -> float:

        if isinstance(data, dict):
            data = Dict(data).to_list()
        elif isinstance(data, tuple):
            data = list(data)
        elif isinstance(data, list):
            data = data.copy()
        else:
            return data

        data = [i for i in data if ValueUtils.is_number(i)]
        if len(data) == 0:
            return None
        else:
            return statistics.mean(data)


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

    # ct windowing (only focus on soft tissue)
    def ct_windowing(ct_img):
        # in origin_dicom, air is -1024. in our ct img, air is 0
        window = 350  # window
        level = 40 + 1024  # level
        high = level + window / 2
        low = level - window / 2
        ct_img = np.where(ct_img > high, high, ct_img)
        ct_img = np.where(ct_img < low, low, ct_img)
        return ct_img

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

    def central_resize(img: ndarray, target_shape: tuple):
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
            # save_nii(cur_cc, "F:/cc_{}.nii".format(segid), NII_SPACING)
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

    # def show(
    #     img: Union[ndarray, Tensor], win_title: str = "", print_info: bool = False
    # ):
    #     if print_info:
    #         print("image data type:", type(img))
    #         print("image shape:", img.shape)
    #         print("image max value:", img.max())
    #         print("image min value:", img.min())

    #     if isinstance(img, Tensor):
    #         # detach: return a tensor share the same memory but without grad
    #         img = img.detach().cpu().numpy()

    #     if len(img.shape) == 3:
    #         img = img[img.shape[0] // 2]
    #     elif len(img.shape) == 4:
    #         img = img[img.shape[0] // 2][img.shape[1] // 2]

    #     cv2.imshow(win_title, img)
    #     cv2.waitKey(0)

    # def save(
    #     img: Union[ndarray, Tensor],
    #     img_name: str = "",
    #     save_dir: str = "",
    #     extension_name: str = ".png",
    # ):
    #     if isinstance(img, Tensor):
    #         # detach: return a tensor share the same memory but without grad
    #         img = img.detach().cpu().numpy()

    #     if len(img.shape) == 3:
    #         img = img[img.shape[0] // 2]
    #     elif len(img.shape) == 4:
    #         img = img[img.shape[0] // 2][img.shape[1] // 2]

    #     if save_dir == "":
    #         save_dir = os.path.join(Global.PROJ_DIR, "debug")
    #     if img_name == "":
    #         img_name = "debug"
    #     if not img_name.endswith(extension_name):
    #         img_name += extension_name
    #     save_path = os.path.join(save_dir, img_name)

    #     imageio.imwrite(save_path, img)
    #     return save_path


class Nii:
    def load(path: str, binary: bool = False, dim: int = 3) -> ndarray:
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        img = img.astype(np.float32)
        if binary:
            img = Img.binarize(img)
        if dim > 0 and len(img.shape) > dim:
            for i in range(len(img.shape) - dim):
                img = np.squeeze(img, axis=0)
        return img

    def save(
        img: Union[ndarray, Tensor],
        save_path: str,
        spacing: tuple = None,
        origin: tuple = None,
        copy_info_from: str = None,
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

        if copy_info_from is not None:
            itk_img.CopyInformation(sitk.ReadImage(copy_info_from))
        else:
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


class File:
    def rename(base_path: str, old_name: str, new_name: str):
        old_path = os.path.join(base_path, old_name)
        new_path = os.path.join(base_path, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            return True
        else:
            return False

    def delete(path: str):
        if os.path.exists(path):
            os.remove(path)
            return True
        else:
            return False

    def get_md5(file_path: str):
        with open(file_path, "rb") as fp:
            md5_obj = hashlib.md5()
            md5_obj.update(fp.read())
            file_md5 = md5_obj.hexdigest()
            # print(file_md5)
            return file_md5

    def get_sha1(file_path: str):
        with open(file_path, "rb") as fp:
            sha1_obj = hashlib.sha1()
            sha1_obj.update(fp.read())
            file_sha1 = sha1_obj.hexdigest()
            # print(file_sha1)
            return file_sha1


class Folder:
    def create(path: str, overwrite: bool = False):
        if overwrite:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        else:
            if not os.path.exists(path):
                os.makedirs(path)
        return path

    def clear(path):
        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
            return True
        else:
            return False

    def rename(base_path: str, old_name: str, new_name: str) -> bool:
        old_path = os.path.join(base_path, old_name)
        new_path = os.path.join(base_path, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            return True
        else:
            return False

    def delete(path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        else:
            return False


class Explorer:
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
        sub_list = Explorer.__get_sub_items(
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
        sub_list = Explorer.__get_sub_items(
            input_dir=input_dir,
            full_path=full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="files",
        )
        return sub_list

    def get_sub_folders(
        input_dir: str,
        key_word: str = "",
        full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ):
        sub_list = Explorer.__get_sub_items(
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
        for input_dir in sub_dirs:
            sub_dirs.extend(Explorer.__walk_sub_dirs(input_dir))
        return sub_dirs

    def walk_sub_dirs(input_dir: str, key_word: str = "", suffle=False) -> List:
        sub_dirs = List()
        for i in Explorer.__walk_sub_dirs(input_dir):
            if key_word == "" or key_word in i:
                sub_dirs.append(i)
        sub_dirs.remove_duplicates()
        if suffle:
            sub_dirs.shuffle()
        else:
            sub_dirs.sort()
        return sub_dirs


class GPU:
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
        else:
            return False

    def used_count() -> int:
        if Global.DEVICE == torch.device("cpu"):
            return 0
        else:
            return torch.cuda.device_count()


class Debug:
    def terminate(err_msg: str):
        print("Error:", err_msg)
        sys.exit(1)

    def clean_debug_data():
        for i in Explorer.walk_sub_dirs(
            Global.TRAIN_RESULTS_DIR, key_word=Global.DELETE_FLAG
        ):
            Folder.delete(i)
        Folder.clear(os.path.join(Global.PROJ_DIR, "debug"))

    def clean_linux_trash():
        if platform.system().lower() == "linux":
            Folder.clear("/home/alan/.local/share/Trash/files/")
            Folder.clear("/home/alan/.local/share/Trash/info/")

    def clean_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
        else:
            return False


class Global:
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    EPS = sys.float_info.epsilon
    DELETE_FLAG = "to.delete"
    METRICS = ["dsc", "msd", "hd95"]

    __settings = Json.load(os.path.join(PROJ_DIR, "settings.json"))

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

    # Windows
    if platform.system().lower() == "windows":
        for i in ["1mm", "3mm"]:
            DATASET_DIR[i] = __settings["dataset.dir.windows.{}".format(i)]
        # window doesn't support pytorch multi-thread
        NUM_WORKERS = 0

    # Linux
    elif platform.system().lower() == "linux":
        for i in ["1mm", "3mm"]:
            DATASET_DIR[i] = __settings["dataset.dir.linux.{}".format(i)]
        NUM_WORKERS = __settings["num.workers"]

    # IMG_SHAPE (Depth, Height, Width)
    IMG_SHAPE = Dict()
    # NII_SPACING (Width, Height, Depth)
    NII_SPACING = Dict()
    for i in ["1mm", "3mm"]:
        IMG_SHAPE[i] = List(__settings["img.shape.{}".format(i)])
        IMG_SHAPE[i] = tuple(int(k) for k in IMG_SHAPE[i])
        NII_SPACING[i] = List(__settings["nii.spacing.{}".format(i)])
        NII_SPACING[i] = tuple(float(k) for k in NII_SPACING[i])

    # Pytorch save/load entire cnn or weight only
    DATASET_FOLDS = __settings["dataset.folds"]
    DATASET_SPLIT_JSON_PATH = os.path.join(PROJ_DIR, __settings["dataset.split.json"])
    HYPER_JSON_PATH_BASELINE = os.path.join(PROJ_DIR, __settings["hyper.json.baseline"])
    HYPER_JSON_PATH_IDL_GTVT = os.path.join(PROJ_DIR, __settings["hyper.json.idl.gtvt"])
    HYPER_JSON_PATH_IDL_GTVN = os.path.join(PROJ_DIR, __settings["hyper.json.idl.gtvn"])
    HYPER_JSON_PATH_IDL_GTVS = os.path.join(PROJ_DIR, __settings["hyper.json.idl.gtvs"])
    TRAIN_RESULTS_DIR = os.path.join(PROJ_DIR, __settings["train.results.dir"])
