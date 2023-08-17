import os
import random
from typing import Tuple

import numpy as np
import torch
from custom import Dict
from custom import Global as g
from custom import Img, Nii
from data_augment import DataAugmentation
from numpy import ndarray
from torch import Tensor


class DataSetBaseline(torch.utils.data.Dataset):
    def __init__(self, patients: list, dataset_ver: str, augment: Dict = None):
        self.__patients = patients
        self.__img_shape = g.IMG_SHAPE[dataset_ver]
        self.__dataset_dir = g.DATASET_DIR[dataset_ver]
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
        img = Img.central_pad_and_crop(img, self.__img_shape)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, 1)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    def get_item(self, patient: str) -> Tuple[Tensor, Tensor]:
        final = Dict()

        # load origin labels
        origin = Img.load_labels(dataset_dir=self.__dataset_dir, patient=patient)

        # loop until target volume is big enough
        tmp = Dict()
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp["seed"] = random.randint(0, 2**16)

            # load gtvs
            tmp["gtvs"] = self.__preprocess(origin["gtvs"], tmp["seed"])
            tmp["gtvs"] = Img.binarize(tmp["gtvs"])

            # target volume is not big enough
            if tmp["gtvs"].sum() < origin["gtvs"].sum() * 0.999:
                # keep the largest gtvs and the augment seed
                if final["gtvs"] == {} or tmp["gtvs"].sum() > final["gtvs"].sum():
                    final["gtvs"] = tmp["gtvs"]
                    final["seed"] = tmp["seed"]
                continue
            # target volume is large enough, break
            else:
                final["gtvs"] = tmp["gtvs"]
                final["seed"] = tmp["seed"]
                break

        # preprocess gtvt and gtvn based on final augment seed
        for gtv in ["gtvt", "gtvn"]:
            final[gtv] = self.__preprocess(origin[gtv], final["seed"])
            final[gtv] = Img.binarize(final[gtv])

        # load background
        background = 1 - torch.maximum(final["gtvt"], final["gtvn"])
        # !!! background FIRST !!!
        labels = torch.cat([background, final["gtvt"], final["gtvn"]], dim=0)

        input_imgs = None
        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(
                self.__dataset_dir, "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = Nii.load(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = Img.ct_windowing(img)

            img = self.__preprocess(img, final["seed"])

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
