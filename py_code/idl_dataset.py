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
        pred_folder: str,
        ignore_other_anotated_slices: bool,
        augment_methods: list = [],
        augment_times: int = 1,
        augment_pct: float = 0.0,
        augment_low_limit: int = 0,
        augment_up_limit: int = 0,
    ):
        self.patient = patient
        self.__label_folder = label_folder
        self.__pred_folder = pred_folder
        self.__ignore_other_anotated_slices = ignore_other_anotated_slices

        self.__augment = DataAugmentation(
            pct=augment_pct,
            methods=augment_methods,
            low_limit=augment_low_limit,
            up_limit=augment_up_limit,
        )

        self.__annotated_slices = NestedDict()
        idx = 0
        for cur_round in reversed(annotated_slices):
            # current step
            for slice_id in annotated_slices[cur_round]:
                for i in range(augment_times):
                    for patch_x in range(3):
                        for patch_y in range(3):
                            self.__annotated_slices[idx]["slice.id"] = slice_id
                            self.__annotated_slices[idx]["patch.x"] = patch_x
                            self.__annotated_slices[idx]["patch.y"] = patch_y
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
        img = img[
            patch_pos[0] : patch_pos[0] + g.PATCH_SIZE[0],
            patch_pos[1] : patch_pos[1] + g.PATCH_SIZE[1],
            patch_pos[2] : patch_pos[2] + g.PATCH_SIZE[2],
        ]
        return img

    def __load_img(
        self,
        img_path: str,
        augment_seed: int,
        patch_pos: tuple,
        weight_map: ndarray,
        binary: bool = False,
    ):
        img = g.load_nii(img_path, binary=binary)

        # make sure img.shape is 3
        for i in range(len(img.shape) - 3):
            img = np.squeeze(img, axis=0)

        # (before augmentation)
        img = g.normalize_img(img)
        # g.show_img(img, "before augment")

        # go through weight map
        img = img * weight_map

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)
        # g.show_img(img, "after augment")

        # !!! NO normalization after augmentation !!!
        # nomalization might give background a positive value when rotating img

        # patch crop after augmentation, max size: 89 283 280
        # a:b -> [a,b)
        img = self.__crop_img(img, patch_pos)
        # g.show_img(img, "patch cropped")

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    # def __remove_non_annotated_slices(self,input_img):

    # must be overrided
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        cur_slice = self.__annotated_slices[idx]

        # (1) simulated iDL
        if self.__label_folder == g.DATASET_FOLDER:
            label_path = os.path.join(
                self.__label_folder, "HNCDL_{}_GTVs.nii".format(self.patient)
            )
        # (2) real iDL
        else:
            pass
            # label_path = os.path.join(self.__label_folder, "????.nii")

        pred_path = os.path.join(self.__pred_folder, "pred_gtvs.nii")

        # get origin pred+label union
        origin_union = g.load_nii(label_path, binary=True)
        origin_union += g.load_nii(pred_path, binary=True)
        origin_union = np.where(origin_union > 1, 1, origin_union)
        # make sure shape is 3
        for i in range(len(origin_union.shape) - 3):
            img = np.squeeze(origin_union, axis=0)

        # weight map
        weight_map = np.zeros_like(origin_union)
        for i in range(weight_map.shape[0]):
            if i == cur_slice:
                weight_map[i] = np.ones_like(weight_map[i])

        # get patch position
        central = ndimage.measurements.center_of_mass(
            origin_union[cur_slice]
        )  # central (h,w)
        patch_pos = [cur_slice]
        patch_pos.append(round(central[0]))
        patch_pos.append(round(central[1]))
        for i in range(3):
            patch_pos[i] -= round(g.PATCH_SIZE[i] / 2)
            if patch_pos[i] <= 0:
                patch_pos[i] = 0
        patch_pos = tuple(patch_pos)

        while 1:
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            augment_seed = random.randint(0, 2**16)

            cropped_union = self.__load_img(
                img_path=label_path,
                augment_seed=augment_seed,
                patch_pos=patch_pos,
                weight_map=weight_map,
                binary=True,
            )
            cropped_union += self.__load_img(
                img_path=pred_path,
                augment_seed=augment_seed,
                patch_pos=patch_pos,
                weight_map=weight_map,
                binary=True,
            )
            cropped_union = torch.where(cropped_union > 1, 1, cropped_union)

            # target volume in the patch is not big enough
            if cropped_union.sum() < origin_union.sum() * 0.99:
                continue
            # target volume in the patch is big enough, break
            else:
                break

        # weight map
        if self.__ignore_other_anotated_slices:
            reserved_slices = [cur_slice]
        else:
            reserved_slices = list(dict.fromkeys(self.__annotated_slices))
        weight_map = np.zeros_like(origin_union)
        for i in range(weight_map.shape[0]):
            if i in reserved_slices:
                weight_map[i] = np.ones_like(weight_map[i])

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
            img_path=label_path,
            augment_seed=augment_seed,
            patch_pos=patch_pos,
            weight_map=weight_map,
        )

        weight_map = self.__crop_img(weight_map, patch_pos)
        weight_map = np.expand_dims(weight_map, axis=0)
        weight_map = torch.from_numpy(weight_map)

        # bg_img = 1 - gtvt_img - gtvn_img
        # g.show_img(bg_img)
        # labels = torch.cat([label_gtvs, label_gtvn, bg_img], dim=0)

        multi_model_imgs = None
        for i in ["CT", "PT", "T1dr", "T2dr"]:
            img_path = os.path.join(
                g.DATASET_FOLDER, "HNCDL_{}_{}.nii".format(self.patient, i)
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

        return multi_model_imgs, label_gtvs, weight_map  # labels


# for testing
# augment_methods = translate / elastic / rotate / scale
annotated_slices = dict()
annotated_slices["round=00"] = [35, 42]
annotated_slices["round=01"] = [24]
pred_folder = os.path.join(
    g.TRAIN_RESULTS_FOLDER,
    "baseline_2022.11.27.06.23.46_target.vol.pct=0_lr=0.0005",
    "baseline",
    "patients",
    "patient=336",
)
tmp_dataset = IDLDataSet(
    patient="336",
    annotated_slices=annotated_slices,
    label_folder=g.DATASET_FOLDER,
    pred_folder=pred_folder,
    ignore_other_anotated_slices=False,
    augment_times=8,
    augment_methods=["rotate"],
    augment_pct=1.0,
    augment_low_limit=1,
    augment_up_limit=1,
)
tmp_dataset.__getitem__(1)
