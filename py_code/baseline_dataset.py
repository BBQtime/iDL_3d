import global_elems as g
import os
import random
import torch
import math
import numpy as np
from nested_dict import NestedDict
from numpy import ndarray
from torch import Tensor
from torchvision import transforms as T
from data_augment import DataAugmentation
from typing import Tuple


class BaselineDataSet(torch.utils.data.Dataset):
    def __init__(self, patient_list: list, augment: dict = {}):

        self.patient_list = patient_list  # make patient_list public

        if len(augment) == 0:
            self.__augment = DataAugmentation()
        else:
            self.__augment = DataAugmentation(
                methods=augment["methods"],
                pct=augment["pct"],
                low_limit=augment["low.limit"],
                up_limit=augment["up.limit"],
            )

    # must be overrided
    def __len__(self):
        return len(self.patient_list)

    def __preprocess(self, img: ndarray, augment_seed: int):
        # DO NOT alter origin img
        img = img.copy()

        # normalize before augmentation
        if not img.max() == img.min() == 0:
            img = g.normalize_img(img)

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # crop and pad after augmentation, max size: 89 283 280
        img = g.central_pad(img, g.IMG_SIZE)
        img = g.central_crop(img, g.IMG_SIZE)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, 1)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    def get_item(self, patient: str) -> Tuple[Tensor, Tensor]:

        origin_gtvs = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)),
            binary=True,
        )
        final_gtvs = None
        final_augment_seed = None

        # loop until target volume is big enough
        for k in range(50):

            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp_augment_seed = random.randint(0, 2**16)

            # load gtvs
            tmp_gtvs = self.__preprocess(origin_gtvs, tmp_augment_seed)
            tmp_gtvs = g.binarize_img(tmp_gtvs)

            # target volume is not big enough
            if tmp_gtvs.sum() < origin_gtvs.sum() * 0.999:
                # keep the largest gtvs and the augment seed
                if final_gtvs is None or tmp_gtvs.sum() > final_gtvs.sum():
                    final_gtvs = tmp_gtvs
                    final_augment_seed = tmp_augment_seed
                continue
            # target volume is large enough, break
            else:
                final_gtvs = tmp_gtvs
                final_augment_seed = tmp_augment_seed
                break

        # load gtvt
        origin_gtvt = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )
        final_gtvt = self.__preprocess(origin_gtvt, final_augment_seed)
        final_gtvt = g.binarize_img(final_gtvt)

        # load gtvn
        gtvn_path = os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(patient))
        if os.path.exists(gtvn_path):
            origin_gtvn = g.load_nii(gtvn_path, binary=True)
        else:
            origin_gtvn = origin_gtvs - origin_gtvt
        final_gtvn = self.__preprocess(origin_gtvn, final_augment_seed)
        final_gtvn = g.binarize_img(final_gtvn)

        # load background
        background = 1 - torch.maximum(final_gtvt, final_gtvn)
        # !!! background FIRST !!!
        labels = torch.cat([background, final_gtvt, final_gtvn], dim=0)

        multi_model_imgs = None
        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = g.load_nii(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = g.ct_windowing(img)

            img = self.__preprocess(img, final_augment_seed)

            # concat multi-model img
            if multi_model_imgs is None:
                multi_model_imgs = img
            else:
                multi_model_imgs = torch.cat([multi_model_imgs, img], dim=0)

        return multi_model_imgs, labels

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.patient_list[idx]
        return self.get_item(patient)


# # for testing
# # augment_methods=[translate / elastic / rotate / scale / flip.lr / flip.ud]
# # patients without GTVn: 257 192
# if 1:
#     tmp_dataset = BaselineDataSet(
#         patient_list=["257"],
#         augment_methods=["rotate"],
#         augment_pct=1,
#         augment_low_limit=1,
#         augment_up_limit=1,
#     )
#     tmp_dataset.__getitem__(0)
