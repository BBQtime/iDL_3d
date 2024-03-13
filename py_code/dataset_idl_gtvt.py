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
from str_lib import Modal, Plane
from torch import Tensor


class DataSetIDLGTVt(DatasetCore):
    def __init__(
        self,
        patient: str,
        selected_slices: Dict,
        pred_dir: str,
        delineation_path: str,  # this is only for observer study
        dataset_ver: str,
        no_pt: bool,
        augment: Dict,
        weight: Dict,
    ):
        super().__init__(dataset_ver=dataset_ver, no_pt=no_pt, augment=augment)
        self.__augment_times = augment["augment.times"]

        # origin images
        self.__origin = Dict()

        # observer study
        if delineation_path is not None:
            self.__origin["label"] = Nii.load(delineation_path, binary=True)
        # simulation
        else:
            self.__origin["label"] = Nii.load(
                os.path.join(
                    g.DATASET_DIR[dataset_ver], "HNCDL_{}_GTVt.nii".format(patient)
                ),
                binary=True,
            )

        # load pred
        self.__origin["pred"] = Nii.load(
            os.path.join(pred_dir, "gtvt_pred.nii.gz"), binary=True
        )

        # load ct/pt/mr1/mr2
        self.__origin[Modal.CT] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_CT.nii".format(patient))
        )
        if not self._no_pt:
            self.__origin[Modal.PT] = Nii.load(
                os.path.join(self._dataset_dir, "HNCDL_{}_PT.nii".format(patient))
            )
        self.__origin[Modal.MR1] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_T1dr.nii".format(patient))
        )
        self.__origin[Modal.MR2] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_T2dr.nii".format(patient))
        )
        # ct windowing
        self.__origin[Modal.CT] = Img.ct_windowing(self.__origin[Modal.CT])

        # load weight map
        self.__origin["weight.map"], selected_slices_mask = self.__load_weight_map(
            selected_slices, weight
        )

        # combine pred(un-selected slices) and label(selected slices)
        self.__origin["label"] *= selected_slices_mask

    def __load_weight_map(self, selected_slices: Dict, weight: Dict):
        # selected slice mask
        selected_slices_mask = Dict()
        max_round = max(
            len(selected_slices[Plane.TRANSVERSE]),
            len(selected_slices[Plane.CORONAL]),
            len(selected_slices[Plane.SAGITTAL]),
        )

        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            # selected slice mask
            selected_slices_mask[plane] = np.zeros(
                self.__origin[Modal.CT].shape, dtype=np.float32
            )

            for round_num in selected_slices[plane]:
                # do NOT change "weight" dict, use another variable
                selected_slice_weight = weight["weight.selected.slice"]
                selected_slice_weight *= pow(
                    weight["weight.prev.round.decay"],
                    (max_round - int(round_num[len("round=") :])),
                )
                if selected_slice_weight < weight["weight.background"]:
                    selected_slice_weight = weight["weight.background"]

                # current step
                for slice_num in selected_slices[plane][round_num]:
                    if plane == Plane.TRANSVERSE:
                        selected_slices_mask[plane][slice_num, :, :] = (
                            np.ones_like(selected_slices_mask[plane][0, :, :])
                            * selected_slice_weight
                        )
                    elif plane == Plane.CORONAL:
                        selected_slices_mask[plane][:, slice_num, :] = (
                            np.ones_like(selected_slices_mask[plane][:, 0, :])
                            * selected_slice_weight
                        )
                    elif plane == Plane.SAGITTAL:
                        selected_slices_mask[plane][:, :, slice_num] = (
                            np.ones_like(selected_slices_mask[plane][:, :, 0])
                            * selected_slice_weight
                        )

        # combine selected_slices_mask on 3 anatomical planes
        selected_slices_mask = np.maximum(
            np.maximum(
                selected_slices_mask[Plane.TRANSVERSE],
                selected_slices_mask[Plane.CORONAL],
            ),
            selected_slices_mask[Plane.SAGITTAL],
        )

        # get fp&fn (keep weight=1 before creating distance map)
        fp = self.__origin["pred"] * (1 - self.__origin["label"])
        fn = (1 - self.__origin["pred"]) * self.__origin["label"]
        fp_plus_fn = fp + fn
        fp_plus_fn = fp_plus_fn * np.where(selected_slices_mask > 0, 1, 0)
        fp_plus_fn = fp_plus_fn.astype(np.float32)

        # pred union label
        pred_union_label = np.maximum(self.__origin["pred"], self.__origin["label"])
        pred_union_label = pred_union_label * np.where(selected_slices_mask > 0, 1, 0)
        pred_union_label = pred_union_label.astype(np.float32)

        # distance map
        if 1:
            distance_map = distance_transform_edt(np.logical_not(pred_union_label))
        else:
            distance_map = distance_transform_edt(np.logical_not(fp_plus_fn))
        distance_map = distance_map.astype(np.float32)
        distance_map = np.where(
            distance_map >= 2 * weight["weight.distance.step"],
            -weight["weight.background"],
            distance_map,
        )
        distance_map = np.where(
            distance_map >= weight["weight.distance.step"],
            -weight["weight.background"] / 2,
            distance_map,
        )
        distance_map = np.where(distance_map >= 0, 0, distance_map)
        distance_map *= -1

        # weighted fp&fn (after weight map)
        fp_plus_fn = (
            fp_plus_fn
            * selected_slices_mask
            * (weight["weight.fp.fn"] / weight["weight.selected.slice"])
        )

        # final_weight_map
        weight_map = np.maximum(
            np.maximum(distance_map, selected_slices_mask), fp_plus_fn
        )

        # return selected_slices_mask to combine pred(un-selected slices) and label(selected slices)
        selected_slices_mask = np.where(selected_slices_mask > 0, 1, 0)

        return weight_map, selected_slices_mask

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

        origin_label_pred_sum = (
            self.__origin["label"].sum() + self.__origin["pred"].sum()
        )

        # loop until target volume is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp["seed"] = random.randint(0, 2**16)

            # load gtvs
            for i in ["label", "pred"]:
                tmp[i] = self._preprocess(
                    img=self.__origin[i], augment_seed=tmp["seed"]
                )
                tmp[i] = Img.binarize(tmp[i])

            tmp_label_pred_sum = tmp["label"].sum() + tmp["pred"].sum()

            # target volume is not large enough
            if tmp_label_pred_sum < origin_label_pred_sum * 0.999:
                # if "final" dict is empty
                if final == {}:
                    for i in ["label", "pred", "seed"]:
                        final[i] = tmp[i]
                    if origin_label_pred_sum == 0:
                        break

                # keep the seed/label/pred with largest target volume
                final_label_pred_sum = final["label"].sum() + final["pred"].sum()
                if tmp_label_pred_sum > final_label_pred_sum:
                    for i in ["label", "pred", "seed"]:
                        final[i] = tmp[i]
                continue

            # target volume is large enough, break
            else:
                for i in ["label", "pred", "seed"]:
                    final[i] = tmp[i]
                break

        # background
        background = 1 - final["label"]
        # !!! background FIRST !!!
        labels = torch.cat([background, final["label"]], dim=0)

        # load multi-modal imgs
        input_imgs = None
        multi_modal_list = [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]
        if self._no_pt:
            multi_modal_list.remove(Modal.PT)
        for i in multi_modal_list:
            img = self._preprocess(self.__origin[i], final["seed"])

            # concat multi-model img
            if input_imgs is None:
                input_imgs = img
            else:
                input_imgs = torch.cat([input_imgs, img], dim=0)

        # weight map
        weight_map = self._preprocess(
            img=self.__origin["weight.map"],
            augment_seed=final["seed"],
            normalize=False,
            clip_up_limit=self.__origin["weight.map"].max(),
        )

        return input_imgs, labels, weight_map
