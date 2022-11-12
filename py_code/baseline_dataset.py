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
from data_augment import DataAugment
from typing import Tuple


class BaselineDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_list: list,
        augment_pct: float = None,
        augment_method: str = None,
        augment_low_limit: int = None,
        augment_up_limit: int = None,
    ):
        self.patient_list = patient_list
        self._init_augment(
            augment_pct=augment_pct,
            augment_method=augment_method,
            augment_low_limit=augment_low_limit,
            augment_up_limit=augment_up_limit,
        )

    def _init_augment(
        self,
        augment_pct: float,
        augment_method: str,
        augment_low_limit: int,
        augment_up_limit: int,
    ):
        if augment_method is not None:
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

    def __load_img(self, img_path: str, augment_seed: int, patch_pos: tuple):
        img = g.load_nii(img_path)

        # make sure img.shape is 3
        for i in range(len(img.shape) - 3):
            img = np.squeeze(img, axis=0)

        # (before augmentation)
        img = g.normalize_img(img)
        g.show_img(img, "before augment")

        # data augmentation
        if self._data_augment is not None:
            img = self._data_augment.run(input_data=img, seed=augment_seed)
        g.show_img(img, "after augment")

        # (no normalize after augmentation)

        # patch crop after augmentation, max size: 89 283 280
        # a:b -> [a,b)
        img = img[
            patch_pos[0] : patch_pos[0] + g.PATCH_SIZE[0],
            patch_pos[1] : patch_pos[1] + g.PATCH_SIZE[1],
            patch_pos[2] : patch_pos[2] + g.PATCH_SIZE[2],
        ]
        # g.show_img(img, "patch cropped")

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    def get_item(
        self,
        patient: str,
        patch_pos: tuple = (),  # make this empty for training
        target_vol_pct: float = 0,  # make this 0 for inference
    ) -> Tuple[Tensor, Tensor]:

        origin_gtvs_img = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient))
        )
        origin_gtvs_img = g.binarize_img(origin_gtvs_img)
        # g.show_img(origin_gtvs_img[20])
        origin_shape = origin_gtvs_img.shape

        while 1:
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            augment_seed = random.randint(0, 2**16)

            # random patch position
            if len(patch_pos) == 0:
                patch_pos = []
                for i in range(3):
                    patch_pos.append(
                        random.randint(0, origin_shape[i] - g.PATCH_SIZE[i])
                    )
                patch_pos = tuple(patch_pos)

            # load gtvt
            # gtvt_path = os.path.join(
            #     g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(cur_patient)
            # )
            # gtvt_img = self.__load_img(img_path=gtvt_path, augment_seed=augment_seed)

            # gtvn_path = os.path.join(
            #     g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(cur_patient)
            # )
            # if os.path.exists(gtvn_path):
            #     gtvn_img = self.__load_img(img_path=gtvn_path, augment_seed=augment_seed)
            # else:
            #     gtvs_path = os.path.join(
            #         g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(cur_patient)
            #     )
            #     gtvs_img = self.__load_img(img_path=gtvs_path, augment_seed=augment_seed)
            #     gtvn_img = gtvs_img - gtvt_img

            # load gtvs
            gtvs_img = self.__load_img(
                img_path=os.path.join(
                    g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)
                ),
                augment_seed=augment_seed,
                patch_pos=patch_pos,
            )

            # target volume is not big enough
            if gtvs_img.sum() < (origin_gtvs_img.sum() * target_vol_pct):
                patch_pos = ()
                continue

            # bg_img = 1 - gtvt_img - gtvn_img
            # g.show_img(bg_img)
            # label_imgs = torch.cat([gtvt_img, gtvn_img, bg_img], dim=0)

            multi_model_imgs = None
            for i in ["CT", "PT", "T1dr", "T2dr"]:
                img_path = os.path.join(
                    g.DATASET_FOLDER, "HNCDL_{}_{}.nii".format(patient, i)
                )
                img = self.__load_img(
                    img_path=img_path,
                    augment_seed=augment_seed,
                    patch_pos=patch_pos,
                )

                # concat multi-model img
                if multi_model_imgs is None:
                    multi_model_imgs = img
                else:
                    multi_model_imgs = torch.cat([multi_model_imgs, img], dim=0)

            break  # target volume is large enough, break

        return multi_model_imgs, gtvs_img  # label_imgs

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.patient_list[idx]
        return self.get_item(patient=patient, patch_pos=(), target_vol_pct=0.5)


# # for testing
# # augment_method= translate / elastic / rotate / scale / combine
# # no gtvn patients: ['138', '152', '168', '174', '175', '192', '194', '204',
# # '220', '229', '239', '247', '257', '261', '276', '309', '311', '315',
# # '323', '327', '333']
# if 1:
#     tmp_dataset = BaselineDataSet(
#         patient_list=["175"],
#         augment_method="combine",
#         augment_pct=1.0,
#         augment_low_limit=2,
#         augment_up_limit=2,
#     )
#     tmp_dataset.__getitem__(0)
