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
    def __init__(
        self,
        patient_list: list,
        augment_methods: list = [],
        augment_pct: float = 0,
        augment_low_limit: int = 0,
        augment_up_limit: int = 0,
    ):
        self.patient_list = patient_list  # make patient_list public

        self.__augment = DataAugmentation(
            methods=augment_methods,
            pct=augment_pct,
            low_limit=augment_low_limit,
            up_limit=augment_up_limit,
        )

    # must be overrided
    def __len__(self):
        return len(self.patient_list)

    def __preprocess(self, img: ndarray, augment_seed: int):
        # normalize before augmentation
        if not img.max() == img.min() == 0:
            img = g.normalize_img(img)

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # crop and pad after augmentation, max size: 89 283 280
        img = g.central_crop(img, g.IMG_SIZE)
        img = g.central_pad(img, g.IMG_SIZE)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    def get_item(
        self,
        patient: str,
        # patch_pos: tuple = (),  # make this empty for training
    ) -> Tuple[Tensor, Tensor]:

        gtvs_path = os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient))
        origin_gtvs = g.load_nii(gtvs_path, binary=True, out_dim=3)
        final_gtvs = None

        # loop until target volume in patch is big enough
        for k in range(50):

            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            augment_seed = random.randint(0, 2**16)

            # load gtvs
            tmp_gtvs = self.__preprocess(origin_gtvs, augment_seed)

            # target volume in the patch is not big enough
            if tmp_gtvs.sum() < origin_gtvs.sum() * 0.999:
                # keep the largest patch
                if final_gtvs is None or tmp_gtvs.sum() > final_gtvs.sum():
                    final_gtvs = tmp_gtvs
                continue
            # target volume is large enough, break
            else:
                final_gtvs = tmp_gtvs
                break

        # load gtvt
        gtvt_path = os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(patient))
        origin_gtvt = g.load_nii(gtvt_path, binary=True, out_dim=3)
        final_gtvt = self.__preprocess(origin_gtvt, augment_seed)

        # load gtvn
        gtvn_path = os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(patient))
        if os.path.exists(gtvn_path):
            origin_gtvn = g.load_nii(gtvn_path, binary=True, out_dim=3)
        else:
            origin_gtvn = origin_gtvs - origin_gtvt
        final_gtvn = self.__preprocess(origin_gtvn, augment_seed)

        # load background
        background = 1 - final_gtvt - final_gtvn

        # !!! background FIRST !!!
        labels = torch.cat([background, final_gtvt, final_gtvn], dim=0)

        multi_model_imgs = None
        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = g.load_nii(nii_path=img_path, out_dim=3)
            img = self.__preprocess(img, augment_seed)

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
