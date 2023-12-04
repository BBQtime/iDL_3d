import os
import random
from typing import Tuple

import numpy as np
import torch
from custom import Dict
from custom import Global as g
from custom import Img, Nii
from dataset_core import DatasetCore
from numpy import ndarray
from scipy.ndimage import distance_transform_edt
from str_lib import (
    AUGMENT_TIMES,
    BACKGROUND,
    CORONAL,
    CT,
    LABEL,
    MR1,
    MR2,
    PRED,
    PT,
    SAGITTAL,
    SEED,
    TRANSVERSE,
    WEIGHT_BACKGROUND,
    WEIGHT_DISTANCE_STEP,
    WEIGHT_FP_FN,
    WEIGHT_MAP,
    WEIGHT_PREV_ROUND_DECAY,
    WEIGHT_SLICE,
)
from torch import Tensor


class DataSetIDLGTVt(DatasetCore):
    def __init__(
        self,
        patient: str,
        selected_slices: Dict,
        pred_dir: str,
        annotation_dir: str,  # this is only for realtime idl
        dataset_ver: str,
        no_pt: bool,
        augment: Dict,
        weight: Dict,
    ):
        super().__init__(dataset_ver=dataset_ver, no_pt=no_pt, augment=augment)
        self.__augment_times = augment[AUGMENT_TIMES]

        # origin images
        self.__origin = Dict()

        # real idl
        if annotation_dir is not None:
            self.__origin[LABEL] = Nii.load(
                os.path.join(annotation_dir, "gtvt_annotation.nii.gz"), binary=True
            )
        # simulation
        else:
            self.__origin[LABEL] = Nii.load(
                os.path.join(
                    g.DATASET_DIR[dataset_ver], "HNCDL_{}_GTVt.nii".format(patient)
                ),
                binary=True,
            )

        # load pred
        self.__origin[PRED] = Nii.load(
            os.path.join(pred_dir, "gtvt_pred.nii.gz"), binary=True
        )

        # load ct/pt/mr1/mr2
        self.__origin[CT] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_CT.nii".format(patient))
        )
        if not self._no_pt:
            self.__origin[PT] = Nii.load(
                os.path.join(self._dataset_dir, "HNCDL_{}_PT.nii".format(patient))
            )
        self.__origin[MR1] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_T1dr.nii".format(patient))
        )
        self.__origin[MR2] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_T2dr.nii".format(patient))
        )
        # ct windowing
        self.__origin[CT] = Img.ct_windowing(self.__origin[CT])

        # load weight map
        self.__origin[WEIGHT_MAP], slice_mask = self.__load_weight_map(
            selected_slices, weight
        )
        # Nii.save(
        #     self.__origin[LABEL],
        #     os.path.join(g.PROJ_DIR, "debug", "annotation.nii.gz"),
        # )
        # Nii.save(
        #     slice_mask,
        #     os.path.join(g.PROJ_DIR, "debug", "slice_mask.nii.gz"),
        # )
        # Nii.save(
        #     self.__origin[WEIGHT_MAP],
        #     os.path.join(g.PROJ_DIR, "debug", "weight_map.nii.gz"),
        # )

        # overwrite pred to label on non-annotated slices
        self.__origin[LABEL] *= slice_mask
        # Nii.save(
        #     self.__origin[LABEL],
        #     os.path.join(g.PROJ_DIR, "debug", "annotation+slice_mask.nii.gz"),
        # )
        # self.__origin[LABEL] += self.__origin[PRED] * (1 - slice_mask)
        # Nii.save(
        #     self.__origin[LABEL],
        #     os.path.join(g.PROJ_DIR, "debug", "annotation+pred.nii.gz"),
        # )

    def __load_weight_map(self, selected_slices: Dict, weight: Dict):
        # annotated slice mask
        slice_mask = Dict()
        max_round = max(
            len(selected_slices[TRANSVERSE]),
            len(selected_slices[CORONAL]),
            len(selected_slices[SAGITTAL]),
        )

        for plane in [TRANSVERSE, CORONAL, SAGITTAL]:
            # annotated slice mask
            slice_mask[plane] = np.zeros(self.__origin[CT].shape, dtype=np.float32)

            for round_num in selected_slices[plane]:
                # do NOT change weight["annotate.slice"], use another variable
                slice_weight = weight[WEIGHT_SLICE]
                slice_weight *= pow(
                    weight[WEIGHT_PREV_ROUND_DECAY],
                    (max_round - int(round_num[len("round=") :])),
                )
                if slice_weight < weight[WEIGHT_BACKGROUND]:
                    slice_weight = weight[WEIGHT_BACKGROUND]

                # current step
                for slice_num in selected_slices[plane][round_num]:
                    if plane == TRANSVERSE:
                        slice_mask[plane][slice_num, :, :] = (
                            np.ones_like(slice_mask[plane][0, :, :]) * slice_weight
                        )
                    elif plane == CORONAL:
                        slice_mask[plane][:, slice_num, :] = (
                            np.ones_like(slice_mask[plane][:, 0, :]) * slice_weight
                        )
                    elif plane == SAGITTAL:
                        slice_mask[plane][:, :, slice_num] = (
                            np.ones_like(slice_mask[plane][:, :, 0]) * slice_weight
                        )

        # combine slice_mask on 3 planes
        slice_mask = np.maximum(
            np.maximum(slice_mask[TRANSVERSE], slice_mask[CORONAL]),
            slice_mask[SAGITTAL],
        )

        # get fp&fn (keep weight=1 before creating distance map)
        fp = self.__origin[PRED] * (1 - self.__origin[LABEL])
        fn = (1 - self.__origin[PRED]) * self.__origin[LABEL]
        fp_plus_fn = fp + fn
        fp_plus_fn = fp_plus_fn * np.where(slice_mask > 0, 1, 0)
        fp_plus_fn = fp_plus_fn.astype(np.float32)

        # pred union label
        pred_union_label = np.maximum(self.__origin[PRED], self.__origin[LABEL])
        pred_union_label = pred_union_label * np.where(slice_mask > 0, 1, 0)
        pred_union_label = pred_union_label.astype(np.float32)

        # distance map
        if 1:
            distance_map = distance_transform_edt(np.logical_not(pred_union_label))
        else:
            distance_map = distance_transform_edt(np.logical_not(fp_plus_fn))
        distance_map = distance_map.astype(np.float32)
        distance_map = np.where(
            distance_map >= 2 * weight[WEIGHT_DISTANCE_STEP],
            -weight[WEIGHT_BACKGROUND],
            distance_map,
        )
        distance_map = np.where(
            distance_map >= weight[WEIGHT_DISTANCE_STEP],
            -weight[WEIGHT_BACKGROUND] / 2,
            distance_map,
        )
        distance_map = np.where(distance_map >= 0, 0, distance_map)
        distance_map *= -1

        # weighted fp&fn (after weight map)
        fp_plus_fn = (
            fp_plus_fn * slice_mask * (weight[WEIGHT_FP_FN] / weight[WEIGHT_SLICE])
        )

        # final_weight_map
        weight_map = np.maximum(np.maximum(distance_map, slice_mask), fp_plus_fn)

        # return slice_mask to overwrite pred to label on non-annotated slices
        slice_mask = np.where(slice_mask > 0, 1, 0)

        return weight_map, slice_mask

    # must be overrided
    def __len__(self):
        return self.__augment_times

    def _preprocess(
        self,
        img: ndarray,
        augment_seed: int,
        normalize: bool = True,  # only for weight map, if normalize is False
        clip_up_limit: float = 1,  # only for weight map, if clip_up_limit is not 1
    ):
        # DO NOT alter origin img
        img = img.copy()

        # normalize before augmentation
        if normalize and (not img.max() == img.min() == 0):
            img = Img.normalize(img)

        # data augmentation
        img = self._augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # pad/crop after augmentation, max size: 89 283 280
        img = Img.central_pad_and_crop(img, self._img_shape)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, clip_up_limit)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    # must be overrided
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        final = Dict()
        tmp = Dict()

        origin_label_pred_sum = self.__origin[LABEL].sum() + self.__origin[PRED].sum()

        # loop until target volume is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp[SEED] = random.randint(0, 2**16)

            # load gtvs
            for i in [LABEL, PRED]:
                tmp[i] = self._preprocess(img=self.__origin[i], augment_seed=tmp[SEED])
                tmp[i] = Img.binarize(tmp[i])

            tmp_label_pred_sum = tmp[LABEL].sum() + tmp[PRED].sum()

            # target volume is not large enough
            if tmp_label_pred_sum < origin_label_pred_sum * 0.999:
                # if "final" dict is empty
                if final == {}:
                    for i in [LABEL, PRED, SEED]:
                        final[i] = tmp[i]
                    if origin_label_pred_sum == 0:
                        break

                # keep the seed/label/pred with largest target volume
                final_label_pred_sum = final[LABEL].sum() + final[PRED].sum()
                if tmp_label_pred_sum > final_label_pred_sum:
                    for i in [LABEL, PRED, SEED]:
                        final[i] = tmp[i]
                continue

            # target volume is large enough, break
            else:
                for i in [LABEL, PRED, SEED]:
                    final[i] = tmp[i]
                break

        # background
        background = 1 - final[LABEL]
        # !!! background FIRST !!!
        labels = torch.cat([background, final[LABEL]], dim=0)

        # load multi-modal imgs
        input_imgs = None
        multi_modal_list = [CT, PT, MR1, MR2]
        if self._no_pt:
            multi_modal_list.remove(PT)
        for i in multi_modal_list:
            img = self._preprocess(self.__origin[i], final[SEED])

            # concat multi-model img
            if input_imgs is None:
                input_imgs = img
            else:
                input_imgs = torch.cat([input_imgs, img], dim=0)

        # weight map
        weight_map = self._preprocess(
            img=self.__origin[WEIGHT_MAP],
            augment_seed=final[SEED],
            normalize=False,
            clip_up_limit=self.__origin[WEIGHT_MAP].max(),
        )

        return input_imgs, labels, weight_map
