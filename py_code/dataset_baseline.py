from custom import Global as g
import os
import random
import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from data_augment import DataAugmentation
from typing import Tuple
from custom import Dict
from custom import Nii
from custom import Img


class DataSetBaseline(torch.utils.data.Dataset):
    def __init__(self, patients: list, augment: Dict = None):
        self.__patients = patients
        self.__augment = DataAugmentation(augment)

    # must be overrided
    def __len__(self):
        return len(self.__patients)

    def __preprocess(self, img: ndarray, augment_seed: int):
        # DO NOT alter origin img
        img = img.copy()

        # normalize before augmentation
        if not img.max() == img.min() == 0:
            img = Img.normalize(img)

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # crop and pad after augmentation, max size: 89 283 280
        img = Img.central_pad(img, g.IMG_SHAPE)
        img = Img.central_crop(img, g.IMG_SHAPE)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, 1)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    def get_item(self, patient: str) -> Tuple[Tensor, Tensor]:
        origin_gtvs = Nii.load(
            os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVs.nii".format(patient)),
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
            tmp_gtvs = Img.binarize(tmp_gtvs)

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
        origin_gtvt = Nii.load(
            os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )
        final_gtvt = self.__preprocess(origin_gtvt, final_augment_seed)
        final_gtvt = Img.binarize(final_gtvt)

        # load gtvn
        origin_gtvn = Nii.load(
            os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVn.nii".format(patient)),
            binary=True,
        )
        final_gtvn = self.__preprocess(origin_gtvn, final_augment_seed)
        final_gtvn = Img.binarize(final_gtvn)

        # load background
        background = 1 - torch.maximum(final_gtvt, final_gtvn)
        # !!! background FIRST !!!
        labels = torch.cat([background, final_gtvt, final_gtvn], dim=0)

        input_imgs = None
        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(g.DATASET_DIR, "HNCDL_{}_{}.nii".format(patient, i))
            img = Nii.load(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = Img.ct_windowing(img)

            img = self.__preprocess(img, final_augment_seed)

            # concat multi-model img
            if input_imgs is None:
                input_imgs = img
            else:
                input_imgs = torch.cat([input_imgs, img], dim=0)

        return input_imgs, labels

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)


# for testing
# augment_methods=[translate / elastic / rotate / scale / flip.lr / flip.ud]
# patients without GTVn: 257 192
if 0:
    augment = Dict()
    # [translate,elastic,rotate,scale,flip.lr,flip.ud]
    augment["methods"] = []
    augment["pct"] = 1
    augment["min"] = 1
    augment["max"] = 1
    augment["times"] = 1
    tmp_dataset = DataSetBaseline(
        patients=["257"],
        augment=augment,
    )
    tmp_dataset.__getitem__(0)
