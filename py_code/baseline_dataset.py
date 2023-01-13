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
        patch_empty_pct: float = -1,  # -1 for valid/test set and inference
        patch_tar_vol_thold: float = 0,  # 0 for valid/test set and inference
    ):
        self.patient_list = patient_list  # make patient_list public

        self.__augment = DataAugmentation(
            methods=augment_methods,
            pct=augment_pct,
            low_limit=augment_low_limit,
            up_limit=augment_up_limit,
        )

        self.__patch_empty_pct = patch_empty_pct
        self.__patch_tar_vol_thold = patch_tar_vol_thold

    # must be overrided
    def __len__(self):
        return len(self.patient_list)

    def __load_img(self, img_path: str, augment_seed: int, patch_pos: tuple):
        img = g.load_nii(nii_path=img_path, out_dim=3)

        # (before augmentation)
        img = g.normalize_img(img)
        # g.save_nii(img, os.path.join(g.PROJ_PATH, "debug", "before_augment.nii"))

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)
        # g.save_nii(img, os.path.join(g.PROJ_PATH, "debug", "after_augment.nii"))

        # no normalization after augmentation,
        # because when rotating img
        # nomalization might give background a positive value

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
    ) -> Tuple[Tensor, Tensor]:

        origin_label_gtvs = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)),
            binary=True,
        )

        patch_empty_thold = random.uniform(0, 1)
        patch_tar_vol_thold_gtvs = origin_label_gtvs.sum() * self.__patch_tar_vol_thold
        label_gtvs = None

        # loop until target volume in patch is big enough
        for attempt_num in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            augment_seed = random.randint(0, 2**16)

            # random patch position
            if len(patch_pos) == 0:
                patch_pos = []
                for i in range(3):
                    patch_pos.append(
                        random.randint(0, origin_label_gtvs.shape[i] - g.PATCH_SIZE[i])
                    )
                patch_pos = tuple(patch_pos)

            # load gtvt
            # gtvt_path = os.path.join(
            #     g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(cur_patient)
            # )
            # gtvt_img = self.__load_img(img_path=gtvt_path, augment_seed=augment_seed)

            # load gtvn
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
            tmp_label_gtvs = self.__load_img(
                img_path=os.path.join(
                    g.DATASET_FOLDER, "HNCDL_{}_GTVs.nii".format(patient)
                ),
                augment_seed=augment_seed,
                patch_pos=patch_pos,
            )

            # current patch must contain label
            if (
                self.__patch_empty_pct == -1
                or self.__patch_empty_pct < patch_empty_thold
            ):
                # target volume in the patch is not big enough
                if tmp_label_gtvs.sum() < patch_tar_vol_thold_gtvs:
                    # keep the largest patch
                    if label_gtvs is None or tmp_label_gtvs.sum() > label_gtvs.sum():
                        label_gtvs = tmp_label_gtvs
                    # print(
                    #     attempt_num,
                    #     label_gtvs.sum() / origin_label_gtvs.sum(),
                    #     tmp_label_gtvs.sum() / origin_label_gtvs.sum(),
                    # )
                    patch_pos = ()
                    continue
                # target volume is large enough, break
                else:
                    label_gtvs = tmp_label_gtvs
                    # print(
                    #     attempt_num,
                    #     label_gtvs.sum() / origin_label_gtvs.sum(),
                    #     tmp_label_gtvs.sum() / origin_label_gtvs.sum(),
                    # )
                    break

            # current patch must be empty
            else:
                # current patch is empty, break
                if tmp_label_gtvs.sum() == 0:
                    label_gtvs = tmp_label_gtvs
                    # print(
                    #     attempt_num,
                    #     label_gtvs.sum() / origin_label_gtvs.sum(),
                    #     tmp_label_gtvs.sum() / origin_label_gtvs.sum(),
                    # )
                    break
                else:
                    # keep the smallest patch
                    if label_gtvs is None or tmp_label_gtvs.sum() < label_gtvs.sum():
                        label_gtvs = tmp_label_gtvs
                    # print(
                    #     attempt_num,
                    #     label_gtvs.sum() / origin_label_gtvs.sum(),
                    #     tmp_label_gtvs.sum() / origin_label_gtvs.sum(),
                    # )
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
        return multi_model_imgs, label_gtvs  # label_imgs

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.patient_list[idx]
        return self.get_item(
            patient=patient,
            patch_pos=(),
        )


# for testing
# augment_methods=[translate / elastic / rotate / scale / flip.lr / flip.ud]
# if 0:
#     tmp_dataset = BaselineDataSet(
#         patient_list=["336"],
#         augment_methods=["rotate"],
#         augment_pct=1,
#         augment_low_limit=1,
#         augment_up_limit=1,
#     )
#     tmp_dataset.__getitem__(0)
