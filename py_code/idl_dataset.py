import global_elems as g
import os
import math
import torch
import numpy as np
from torch import Tensor
from typing import Tuple
import random
from data_augment import DataAugmentation
from scipy import ndimage
from numpy import ndarray
from nested_dict import NestedDict


class IDLDataSet:
    def __init__(
        self,
        patient: str,
        annotated_slices: dict,
        label_folder: str,
        ignore_other_anotated_slices: bool,
        augment_methods: list = [],
        augment_times: int = 1,
        augment_pct: float = 0.0,
        augment_low_limit: int = 0,
        augment_up_limit: int = 0,
    ):
        self.patient = patient
        self.__ignore_other_anotated_slices = ignore_other_anotated_slices
        self.__augment = DataAugmentation(
            pct=augment_pct,
            methods=augment_methods,
            low_limit=augment_low_limit,
            up_limit=augment_up_limit,
        )

        # (1) simulated iDL
        if label_folder == g.DATASET_FOLDER:
            self.__gtvs_path = os.path.join(
                label_folder, "HNCDL_{}_GTVs.nii".format(self.patient)
            )
        # (2) real iDL
        else:
            self.__gtvs_path = os.path.join(label_folder, "????.nii")

        # get patch position
        self.__img_shape = g.load_nii(self.__gtvs_path, out_dim=3).shape

        self.__annotated_slices = NestedDict()
        idx = 0
        for cur_round in reversed(annotated_slices):
            # current step
            for cur_slice in annotated_slices[cur_round]:
                d = cur_slice - int(g.PATCH_SIZE[0] / 2)
                d = g.check_limit(d, 0, self.__img_shape[0] - g.PATCH_SIZE[0])
                # augmentation times
                for times in range(augment_times):
                    # for h in range(3):
                    for h in [1]:
                        # for w in range(3):
                        for w in [1]:
                            patch_pos = [d, h, w]
                            for i in [1, 2]:
                                if patch_pos[i] == 2:
                                    patch_pos[i] = self.__img_shape[i] - g.PATCH_SIZE[i]
                                elif patch_pos[i] == 1:
                                    patch_pos[i] = round(
                                        (self.__img_shape[i] - g.PATCH_SIZE[i]) / 2
                                    )
                                    patch_pos[i] = g.check_limit(
                                        patch_pos[i],
                                        0,
                                        self.__img_shape[i] - g.PATCH_SIZE[i],
                                    )
                            self.__annotated_slices[idx]["slice.id"] = cur_slice
                            self.__annotated_slices[idx]["patch.pos"] = tuple(patch_pos)
                            idx += 1
            if augment_times >= 16:
                augment_times /= 4
            else:
                augment_times /= 2
            # rounded up
            augment_times = math.ceil(augment_times)
        # print(self.__annotated_slices)

    # must be overrided
    def __len__(self):
        return len(self.__annotated_slices)

    def __crop_img(self, img, patch_pos):
        # a:b means [a,b)
        img = img[
            patch_pos[0] : patch_pos[0] + g.PATCH_SIZE[0],
            patch_pos[1] : patch_pos[1] + g.PATCH_SIZE[1],
            patch_pos[2] : patch_pos[2] + g.PATCH_SIZE[2],
        ]
        return img

    def __load_weight_map(self, cur_slice, patch_pos):
        if self.__ignore_other_anotated_slices:
            reserved_slices = [cur_slice]
        else:
            reserved_slices = []
            for i in self.__annotated_slices:
                reserved_slices.append(self.__annotated_slices[i]["slice.id"])
            reserved_slices = g.list_remove_duplicates(reserved_slices)

        weight_map = np.zeros(self.__img_shape, dtype=np.float32)
        for i in range(weight_map.shape[0]):
            if i in reserved_slices:
                weight_map[i] = np.ones_like(weight_map[i])

        # g.save_nii(
        #     weight_map, os.path.join(g.PROJ_PATH, "debug", "weight_map_origin.nii")
        # )
        weight_map = self.__crop_img(weight_map, patch_pos)
        # g.save_nii(
        #     weight_map, os.path.join(g.PROJ_PATH, "debug", "weight_map_patch.nii")
        # )

        # unsqueeze img to 4 dim before convert to Tensor
        weight_map = np.expand_dims(weight_map, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        weight_map = torch.from_numpy(weight_map)
        return weight_map

    def __load_img(
        self,
        img_path: str,
        augment_seed: int,
        patch_pos: tuple,
    ):
        # make sure img.shape is 3
        img = g.load_nii(img_path, binary=False, out_dim=3)

        # nomalization before augmentation
        img = g.normalize_img(img)
        # g.show_img(img, "before augment")

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)
        # g.show_img(img, "after augment")

        # !!! NO normalization after augmentation !!!
        # nomalization might give background a positive value when rotating img

        # patch crop after augmentation, max size: 89 283 280
        # weight_map = g.load_nii(
        #     os.path.join(g.PROJ_PATH, "debug", "weight_map_origin.nii")
        # )
        # g.save_nii(
        #     img * weight_map, os.path.join(g.PROJ_PATH, "debug", "img_origin.nii")
        # )
        img = self.__crop_img(img, patch_pos)
        # weight_map = g.load_nii(
        #     os.path.join(g.PROJ_PATH, "debug", "weight_map_patch.nii")
        # )
        # g.save_nii(
        #     img * weight_map, os.path.join(g.PROJ_PATH, "debug", "img_patch.nii")
        # )

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    # must be overrided
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        cur_slice = self.__annotated_slices[idx]["slice.id"]
        patch_pos = self.__annotated_slices[idx]["patch.pos"]
        weight_map = self.__load_weight_map(cur_slice=cur_slice, patch_pos=patch_pos)

        # make sure same group use the same augment_seed
        # !!! use python random, DO NOT use np.random !!!
        # np.random + dataloader will cause multi-processing problem
        augment_seed = random.randint(0, 2**16)

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

        label_gtvs = self.__load_img(
            img_path=self.__gtvs_path, augment_seed=augment_seed, patch_pos=patch_pos
        )

        # bg_img = 1 - gtvt_img - gtvn_img
        # g.show_img(bg_img)
        # labels = torch.cat([label_gtvs, label_gtvn, bg_img], dim=0)

        multi_model_imgs = None
        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_{}.nii".format(self.patient, i)
            )
            img = self.__load_img(
                img_path=img_path, augment_seed=augment_seed, patch_pos=patch_pos
            )

            # concat multi-model img
            if multi_model_imgs is None:
                multi_model_imgs = img
            else:
                multi_model_imgs = torch.cat([multi_model_imgs, img], dim=0)

        return multi_model_imgs, label_gtvs, weight_map  # labels


# for testing
# augment_methods = translate / elastic / rotate / scale
if 0:
    annotated_slices = dict()
    annotated_slices["round=00"] = [20, 40]
    annotated_slices["round=01"] = [30]
    # pred_folder = os.path.join(
    #     g.TRAIN_RESULTS_FOLDER,
    #     "baseline_2022.11.27.06.23.46_target.vol.pct=0_lr=0.0005",
    #     "baseline",
    #     "patients",
    #     "patient=336",
    # )
    tmp_dataset = IDLDataSet(
        patient="336",
        annotated_slices=annotated_slices,
        label_folder=g.DATASET_FOLDER,
        # pred_folder=pred_folder,
        ignore_other_anotated_slices=False,
        augment_times=1,
        augment_methods=[],  # ["rotate"],
        augment_pct=1.0,
        augment_low_limit=1,
        augment_up_limit=1,
    )
    tmp_dataset.__getitem__(2)
