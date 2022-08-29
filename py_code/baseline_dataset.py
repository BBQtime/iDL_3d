import global_elems as g
import os
import random
import torch
import numpy as np
from nested_dict import NestedDict
from numpy import ndarray
from torch import Tensor
from torchvision import transforms as T
from data_augment import DataAugment


class BaselineDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_list: list,
        augment_pct: float = 0.0,
        augment_method: str = None,
        augment_low_limit: int = 1,
        augment_up_limit: int = 1,
    ):
        self.patient_list = patient_list
        self._init_augment(
            augment_pct=augment_pct,
            augment_method=augment_method,
            augment_low_limit=augment_low_limit,
            augment_up_limit=augment_up_limit,
        )
        # patient_slice_mapping is a list of: ["patient", "slice"]
        # self._init_patient_slice_mapping(self.patient_list)

    def _init_augment(
        self,
        augment_pct: float = 0.0,
        augment_method: str = None,
        augment_low_limit: int = 1,
        augment_up_limit: int = 1,
    ):
        if augment_pct > 0.0 and augment_method is not None:
            self._data_augment = DataAugment(
                pct=augment_pct,
                method=augment_method,
                low_limit=augment_low_limit,
                up_limit=augment_up_limit,
            )
        else:
            self._data_augment = None

    # must be overrided
    def __len__(self):
        return len(self.patient_list)

    # max size: 89 283 280
    def __pad_and_crop(self, img: ndarray) -> ndarray:
        # step 1: padding
        in_size = NestedDict()
        in_size["d"], in_size["h"], in_size["w"] = img.shape

        out_size = NestedDict()
        out_size["d"] = g.IMG_SIZE[2]
        out_size["h"] = g.IMG_SIZE[1]
        out_size["w"] = g.IMG_SIZE[0]

        pad = NestedDict()
        for i in ["w", "h", "d"]:
            pad[i][0] = pad[i][1] = 0

        for i in ["w", "h", "d"]:
            if out_size[i] > in_size[i]:
                cur_pad = out_size[i] - in_size[i]
                if cur_pad % 2 == 0:
                    pad[i][0] = pad[i][1] = int(cur_pad / 2)
                else:
                    pad[i][0] = int(cur_pad / 2)
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

        # step 2: cropping
        in_size["d"], in_size["h"], in_size["w"] = img.shape

        if (
            in_size["d"] > out_size["d"]
            or in_size["h"] > out_size["h"]
            or in_size["w"] > out_size["w"]
        ):
            start_point = NestedDict()

            for i in ["w", "h", "d"]:
                start_point[i] = (in_size[i] // 2) - (out_size[i] // 2)

            img = img[
                start_point["d"] : start_point["d"] + out_size["d"],
                start_point["h"] : start_point["h"] + out_size["h"],
                start_point["w"] : start_point["w"] + out_size["w"],
            ]
        return img

    def __load_img(self, img_path: str, augment_seed: int):
        img = g.load_nii(img_path).astype(np.float32)

        # make sure img.shape is 3
        for i in range(len(img.shape) - 3):
            img = np.squeeze(img, axis=0)

        img = self.__pad_and_crop(img)

        if self._data_augment is not None:
            img = self._data_augment.run(input_data=img, seed=augment_seed)

        # unsqueeze img to 4 dim
        img = np.expand_dims(img, axis=0)

        # numpy to tensor
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    def _get_item(
        self, cur_patient: str, gtvt_path: str, gtvn_path: str
    ) -> tuple[Tensor, Tensor]:

        # make sure same group use the same augment seed
        # !!! use python random, do not use np.random !!!
        # np.random + dataloader will cause multi-processing problem
        augment_seed = random.randint(0, 2 ** 16)

        multi_model_imgs = None

        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_{}.nii".format(cur_patient, i)
            )
            img = self.__load_img(img_path=img_path, augment_seed=augment_seed)
            # g.show_img(img)

            # concat multi-model img
            if multi_model_imgs is None:
                multi_model_imgs = img
            else:
                multi_model_imgs = torch.cat([multi_model_imgs, img], dim=0)

        # load gtvt
        gtvt_img = self.__load_img(img_path=gtvt_path, augment_seed=augment_seed)
        # g.show_img(gtvt_img)

        # load gtvn
        if os.path.exists(gtvn_path):
            gtvn_img = self.__load_img(img_path=gtvn_path, augment_seed=augment_seed)
        else:
            gtvs_path = g.change_char_in_str(gtvn_path, -5, "s")
            gtvs_img = self.__load_img(img_path=gtvs_path, augment_seed=augment_seed)
            gtvn_img = gtvs_img - gtvt_img
            # g.show_img(gtvs_img)
            # g.show_img(gtvt_img)
            # g.show_img(gtvn_img)

        bg_img = 1 - gtvt_img - gtvn_img
        # g.show_img(bg_img)
        label_imgs = torch.cat([gtvt_img, gtvn_img, bg_img], dim=0)

        return multi_model_imgs, label_imgs

    # must be overrided
    def __getitem__(self, idx: int):
        cur_patient = self.patient_list[idx]
        # cur_slice = self.patient_slice_mapping[idx][1]
        # cur_slice_folder = os.path.join(g.DATASET_FOLDER, cur_patient, cur_slice)
        gtvt_path = os.path.join(
            g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(cur_patient)
        )
        gtvn_path = os.path.join(
            g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(cur_patient)
        )
        return self._get_item(
            cur_patient=cur_patient, gtvt_path=gtvt_path, gtvn_path=gtvn_path
        )


# for testing
# augment_method= translate / elastic / rotate / scale / combine
# no gtvn patients: ['138', '152', '168', '174', '175', '192', '194', '204',
# '220', '229', '239', '247', '257', '261', '276', '309', '311', '315',
# '323', '327', '333']
if 1:
    tmp_dataset = BaselineDataSet(
        patient_list=["175"],
        augment_method="combine",
        augment_pct=1.0,
        augment_low_limit=2,
        augment_up_limit=2,
    )
    tmp_dataset.__getitem__(0)
