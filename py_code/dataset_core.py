import global_core as g
import numpy as np
import torch
from custom_dict import Dict
from data_augment import DataAugmentation
from numpy import ndarray


class DatasetCore(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_ver: str,
        no_pt: bool,
        augment: Dict = None,
    ):
        self._dataset_dir = g.DATASET_DIR[dataset_ver]
        self._img_shape = g.IMG_SHAPE
        self._no_pt = no_pt
        self._augment = DataAugmentation(param=augment)

    def _preprocess(self, img: ndarray, augment_seed: int):
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
