import os
import sys
import warnings
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
# after json loaded, key(int type) will transfrom into string
# better make all keys "string" type
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
        return List(identical_items)

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

    def sort(self):
        super().__init__(natsorted(self))

    def remove_duplicates(self):
        self[:] = List(set(self))


class Value:
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

    def get_avg(input_data: Union[list, dict]) -> float:
        if isinstance(input_data, list):
            return sum(input_data) / len(input_data)

        elif isinstance(input_data, dict):
            input_data = Dict(input_data.copy()).to_list()
            return sum(input_data) / len(input_data)

        else:
            return input_data

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
    def central_crop(img: ndarray, shape: tuple) -> ndarray:
        in_shape = Dict()
        in_shape["d"], in_shape["h"], in_shape["w"] = img.shape

        out_shape = Dict()
        out_shape["d"] = shape[0]
        out_shape["h"] = shape[1]
        out_shape["w"] = shape[2]

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

    def central_pad(img: ndarray, shape: tuple) -> ndarray:
        in_shape = Dict()
        in_shape["d"], in_shape["h"], in_shape["w"] = img.shape

        out_shape = Dict()
        out_shape["d"] = shape[0]
        out_shape["h"] = shape[1]
        out_shape["w"] = shape[2]

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
    #     save_folder: str = "",
    #     extension_name: str = ".png",
    # ):
    #     if isinstance(img, Tensor):
    #         # detach: return a tensor share the same memory but without grad
    #         img = img.detach().cpu().numpy()

    #     if len(img.shape) == 3:
    #         img = img[img.shape[0] // 2]
    #     elif len(img.shape) == 4:
    #         img = img[img.shape[0] // 2][img.shape[1] // 2]

    #     if save_folder == "":
    #         save_folder = os.path.join(Global.PROJ_PATH, "debug")
    #     if img_name == "":
    #         img_name = "debug"
    #     if not img_name.endswith(extension_name):
    #         img_name += extension_name
    #     save_path = os.path.join(save_folder, img_name)

    #     imageio.imwrite(save_path, img)
    #     return save_path


class Nii:
    def load(
        nii_path: str,
        binary: bool = False,
        out_dim: int = 3,
    ) -> ndarray:
        img = sitk.ReadImage(nii_path)
        img = sitk.GetArrayFromImage(img)
        img = img.astype(np.float32)
        if binary:
            img = Img.binarize(img)
        if out_dim > 0 and len(img.shape) > out_dim:
            for i in range(len(img.shape) - out_dim):
                img = np.squeeze(img, axis=0)
        return img

    def save(img: Union[ndarray, Tensor], save_path: str, spacing: tuple = ()):
        if isinstance(img, Tensor):
            # detach: return a tensor share the same memory but without grad
            img = img.detach().cpu().numpy()

        if len(img.shape) > 3:
            for i in range(len(img.shape) - 3):
                if img.shape[i] == 1:
                    img = np.squeeze(img, axis=0)
                else:
                    img = img[0]

        itk_img = sitk.GetImageFromArray(img)
        if spacing == ():
            itk_img.SetSpacing(Global.NII_SPACING)
        else:
            itk_img.SetSpacing(spacing)
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
        folder_path: str,
        return_full_path: bool,
        key_word: str,
        shuffle: bool,
        seed: int,
        select: str,
    ) -> List:

        sub_list = List(os.listdir(folder_path))

        if select != "both":
            for sub_name in sub_list.copy():
                if select == "files":
                    if not os.path.isfile(os.path.join(folder_path, sub_name)):
                        sub_list.remove(sub_name)
                elif select == "folders":
                    if os.path.isfile(os.path.join(folder_path, sub_name)):
                        sub_list.remove(sub_name)

        if shuffle:
            sub_list.shuffle(seed)
        else:
            sub_list.sort()

        if key_word != "":
            for i in sub_list.copy():
                if key_word not in i:
                    sub_list.remove(i)

        if return_full_path:
            for i in range(len(sub_list)):
                sub_list[i] = os.path.join(folder_path, sub_list[i])

        return sub_list

    def get_sub_items(
        folder_path: str,
        key_word: str = "",
        return_full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ) -> List:
        sub_list = Explorer.__get_sub_items(
            folder_path=folder_path,
            return_full_path=return_full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="both",
        )
        return sub_list

    def get_sub_files(
        folder_path: str,
        key_word: str = "",
        return_full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ) -> List:
        sub_list = Explorer.__get_sub_items(
            folder_path=folder_path,
            return_full_path=return_full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="files",
        )
        return sub_list

    def get_sub_folders(
        folder_path: str,
        key_word: str = "",
        return_full_path: bool = False,
        shuffle: bool = False,
        seed: int = None,
    ):
        sub_list = Explorer.__get_sub_items(
            folder_path=folder_path,
            return_full_path=return_full_path,
            key_word=key_word,
            shuffle=shuffle,
            seed=seed,
            select="folders",
        )
        return sub_list

    def __walk_sub_folders(folder_path: str) -> List:
        sub_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for folder_path in sub_folders:
            sub_folders.extend(Explorer.__walk_sub_folders(folder_path))
        return sub_folders

    def walk_sub_folders(folder_path: str, key_word: str = "") -> List:
        sub_folders = List()
        for i in Explorer.__walk_sub_folders(folder_path):
            if key_word == "" or key_word in i:
                sub_folders.append(i)
        return sub_folders


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


class Cleaner:
    def clean_debug_data():
        for cur_folder in Explorer.walk_sub_folders(
            Global.TRAIN_RESULTS_FOLDER, key_word="delete.this"
        ):
            Folder.delete(cur_folder)
        Folder.clear(os.path.join(Global.PROJ_PATH, "debug"))

    def clean_linux_trash():
        if platform.system().lower() == "linux":
            Folder.clear("/home/alan/.local/share/Trash/files/")
            Folder.clear("/home/alan/.local/share/Trash/info/")


class Global:
    PROJ_PATH = os.path.dirname(os.path.dirname(__file__))
    EPS = sys.float_info.epsilon
    METRICS = ["dsc", "msd", "hd95"]

    __settings = Json.load(os.path.join(PROJ_PATH, "settings.json"))

    # use CPU
    if __settings["use.gpu"] is False:
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

    # Windows or Linux
    if platform.system().lower() == "windows":
        DATASET_FOLDER = __settings["dataset.folder.win"]
        NUM_WORKERS = 0  # window doesn't support pytorch multi-thread

    elif platform.system().lower() == "linux":
        DATASET_FOLDER = __settings["dataset.folder.linux"]
        NUM_WORKERS = __settings["num.workers"]

    # Depth, Height, Width
    IMG_SHAPE = List(__settings["img.shape"])
    IMG_SHAPE = tuple(int(i) for i in IMG_SHAPE)

    # Width, Height, Depth
    NII_SPACING = List(__settings["nii.spacing"])
    NII_SPACING = tuple(float(i) for i in NII_SPACING)

    # Pytorch save/load entire cnn or weight only
    CNN_STATE_DICT_ONLY = __settings["cnn.state.dict.only"]
    MAX_BATCH_SIZE_PER_GPU = __settings["max.batch.size.per.gpu"]
    DATASET_K_FOLDS = __settings["dataset.k.folds"]
    DATASET_SPLIT_JSON = os.path.join(PROJ_PATH, __settings["dataset.split.json"])
    HYPER_JSON_BASELINE = os.path.join(PROJ_PATH, __settings["hyper.json.baseline"])
    HYPER_JSON_IDL_GTVT = os.path.join(PROJ_PATH, __settings["hyper.json.idl.gtvt"])
    HYPER_JSON_IDL_GTVN = os.path.join(PROJ_PATH, __settings["hyper.json.idl.gtvn"])
    TRAIN_RESULTS_FOLDER = os.path.join(PROJ_PATH, __settings["train.results.folder"])
