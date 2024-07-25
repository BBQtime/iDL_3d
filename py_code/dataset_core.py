import os

import global_core as g
import numpy as np
import torch
from custom_dict import Dict
from data_augment import DataAugmentation
from numpy import ndarray
from str_lib import DatasetVer, ErrMsg, Modal
from torch import Tensor


class DatasetCore(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_ver: str,
        no_pt: bool,
        augment: Dict = None,
    ):
        self._dataset_ver = dataset_ver
        self._img_shape = g.IMG_SHAPE
        self._no_pt = no_pt
        self._augment = DataAugmentation(param=augment)

    def _load_multi_modal_imgs(
        self,
        dataset_ver: str,
        patient: str,
        no_pt: bool,
    ):
        img_path = Dict()
        img_path[Modal.CT] = "CT"
        if dataset_ver == DatasetVer.NKI:
            img_path[Modal.PT] = "PTdr"
        else:
            img_path[Modal.PT] = "PT"
        img_path[Modal.MR1] = "T1dr"
        img_path[Modal.MR2] = "T2dr"

        multi_modal_imgs = Dict()

        for i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
            if i == Modal.PT and no_pt:
                continue

            if dataset_ver in [DatasetVer.AU, DatasetVer.OBS_STUDY]:
                img_path[i] = "HNCDL_{}_{}.nii".format(patient, img_path[i])
                img_path[i] = os.path.join(g.DATASET_DIR[dataset_ver], img_path[i])

            elif dataset_ver in [DatasetVer.MDA, DatasetVer.NKI]:
                img_path[i] = "{}_{}.nii".format(patient, img_path[i])
                img_path[i] = os.path.join(
                    g.DATASET_DIR[dataset_ver], patient, img_path[i]
                )

            else:
                g.error_exit(ErrMsg.DATASET_VER_INVALID)

            multi_modal_imgs[i] = g.load_nii(img_path[i])

            # windowing
            if i == Modal.CT:
                multi_modal_imgs[i] = g.windowing_ct(multi_modal_imgs[i])

        return multi_modal_imgs

    def _preprocess(self, img: ndarray, augment_seed: int) -> Tensor:
        # DO NOT alter origin img
        img = img.copy()

        # normalize before augmentation
        if not img.max() == img.min() == 0:
            img = g.normalize_img(img)

        # data augmentation
        img = self._augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # crop and pad after augmentation, max size: 89 283 280
        img = g.center_align_img(img, self._img_shape)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, 1)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img
