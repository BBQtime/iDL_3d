import os
import random
from typing import Tuple

import numpy as np
import torch
from custom import Dict, Img, Nii
from dataset_core import DatasetCore
from numpy import ndarray
from scipy.ndimage import distance_transform_edt
from str_lib import StrLib as s
from torch import Tensor


class DataSetIDLGTVt(DatasetCore):
    def __init__(
        self,
        patient: str,
        selected_slices: Dict,
        label_dir: str,  # dont use g.DATASET_DIR because of realtime training
        pred_dir: str,
        dataset_ver: str,
        no_pt: bool,
        augment: Dict,
        weight: Dict,
    ):
        super().__init__(dataset_ver=dataset_ver, no_pt=no_pt, augment=augment)
        self.__augment_times = augment["times"]

        # origin images
        self.__origin = Dict()

        # simulated iDL label path
        label_path = os.path.join(label_dir, "HNCDL_{}_GTVt.nii".format(patient))
        if not os.path.exists(label_path):
            # real iDL label path
            label_path = os.path.join(label_dir, "gtvt_label.nii")

        # load label
        self.__origin[s.LABEL] = Nii.load(label_path, binary=True)

        # load pred
        self.__origin[s.PRED] = Nii.load(
            os.path.join(pred_dir, "gtvt_pred.nii"), binary=True
        )

        # load ct/pt/mr1/mr2
        self.__origin[s.CT] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_CT.nii".format(patient))
        )
        if not self._no_pt:
            self.__origin[s.PT] = Nii.load(
                os.path.join(self._dataset_dir, "HNCDL_{}_PT.nii".format(patient))
            )
        self.__origin[s.MR1] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_T1dr.nii".format(patient))
        )
        self.__origin[s.MR2] = Nii.load(
            os.path.join(self._dataset_dir, "HNCDL_{}_T2dr.nii".format(patient))
        )
        # ct windowing
        self.__origin[s.CT] = Img.ct_windowing(self.__origin[s.CT])

        # load weight map
        self.__origin["weight.map"], slice_mask = self.__load_weight_map(
            selected_slices, weight
        )

        # save origin label and pred
        # Nii.save(
        #     self.__origin[s.LABEL],
        #     os.path.join(g.PROJ_DIR, "debug", "origin_label.nii"),
        # )
        # Nii.save(
        #     self.__origin[s.PRED], os.path.join(g.PROJ_DIR, "debug", "pred.nii")
        # )

        # overwrite pred to label on non-annotated slices
        self.__origin[s.LABEL] *= slice_mask
        self.__origin[s.LABEL] += self.__origin[s.PRED] * (1 - slice_mask)

        # save overwrited label
        # Nii.save(
        #     self.__origin[s.LABEL],
        #     os.path.join(g.PROJ_DIR, "debug", "overwrite_label.nii"),
        # )

    def __load_weight_map(self, selected_slices: Dict, weight: Dict):
        # annotated slice mask
        slice_mask = Dict()
        max_round = max(
            len(selected_slices[s.TRANSVERSE]),
            len(selected_slices[s.CORONAL]),
            len(selected_slices[s.SAGITTAL]),
        )

        for plane in [s.TRANSVERSE, s.CORONAL, s.SAGITTAL]:
            # annotated slice mask
            slice_mask[plane] = np.zeros(self.__origin[s.CT].shape, dtype=np.float32)

            for round_num in selected_slices[plane]:
                # do NOT change weight["annotate.slice"], use another variable
                slice_weight = weight["slice"]
                slice_weight *= pow(
                    weight["prev.round.decay"],
                    (max_round - int(round_num[len("round=") :])),
                )
                if slice_weight < weight[s.BACKGROUND]:
                    slice_weight = weight[s.BACKGROUND]

                # current step
                for slice_num in selected_slices[plane][round_num]:
                    if plane == s.TRANSVERSE:
                        slice_mask[plane][slice_num, :, :] = (
                            np.ones_like(slice_mask[plane][0, :, :]) * slice_weight
                        )
                    elif plane == s.CORONAL:
                        slice_mask[plane][:, slice_num, :] = (
                            np.ones_like(slice_mask[plane][:, 0, :]) * slice_weight
                        )
                    elif plane == s.SAGITTAL:
                        slice_mask[plane][:, :, slice_num] = (
                            np.ones_like(slice_mask[plane][:, :, 0]) * slice_weight
                        )

        # combine slice_mask on 3 planes
        slice_mask = np.maximum(
            np.maximum(slice_mask[s.TRANSVERSE], slice_mask[s.CORONAL]),
            slice_mask[s.SAGITTAL],
        )
        # Nii.save(slice_mask, os.path.join(g.PROJ_DIR, "debug", "slice_mask.nii"))

        # get fp&fn (keep weight=1 before creating distance map)
        fp = self.__origin[s.PRED] * (1 - self.__origin[s.LABEL])
        fn = (1 - self.__origin[s.PRED]) * self.__origin[s.LABEL]
        fp_fn = fp + fn
        fp_fn = fp_fn * np.where(slice_mask > 0, 1, 0)
        fp_fn = fp_fn.astype(np.float32)

        # annotation (pred + label)
        if 1:
            annotation = np.maximum(self.__origin[s.PRED], self.__origin[s.LABEL])
            annotation = annotation * np.where(slice_mask > 0, 1, 0)
            annotation = annotation.astype(np.float32)
            # Nii.save(annotation, os.path.join(g.PROJ_DIR, "debug", "annotation.nii"))

        # distance map
        if 1:
            distance_map = distance_transform_edt(np.logical_not(annotation))
        else:
            distance_map = distance_transform_edt(np.logical_not(fp_fn))
        distance_map = distance_map.astype(np.float32)
        distance_map = np.where(
            distance_map >= 2 * weight[s.DISTANCE_STEP],
            -weight[s.BACKGROUND],
            distance_map,
        )
        distance_map = np.where(
            distance_map >= weight[s.DISTANCE_STEP],
            -weight[s.BACKGROUND] / 2,
            distance_map,
        )
        distance_map = np.where(distance_map >= 0, 0, distance_map)
        distance_map *= -1
        # Nii.save(distance_map, os.path.join(g.PROJ_DIR, "debug", "distance_map.nii"))

        # weighted fp&fn (after weight map)
        fp_fn = fp_fn * slice_mask * (weight["fp.fn"] / weight["slice"])
        # Nii.save(fp_fn, os.path.join(g.PROJ_DIR, "debug", "fp_fn.nii"))

        # final_weight_map
        weight_map = np.maximum(np.maximum(distance_map, slice_mask), fp_fn)
        # Nii.save(weight_map, os.path.join(g.PROJ_DIR, "debug", "weight_map.nii"))

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

        origin_label_pred_sum = (
            self.__origin[s.LABEL].sum() + self.__origin[s.PRED].sum()
        )

        # loop until target volume is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp[s.SEED] = random.randint(0, 2**16)

            # load gtvs
            for i in [s.LABEL, s.PRED]:
                tmp[i] = self._preprocess(
                    img=self.__origin[i], augment_seed=tmp[s.SEED]
                )
                tmp[i] = Img.binarize(tmp[i])

            tmp_label_pred_sum = tmp[s.LABEL].sum() + tmp[s.PRED].sum()

            # target volume is not large enough
            if tmp_label_pred_sum < origin_label_pred_sum * 0.999:
                # if "final" dict is empty
                if final == {}:
                    for i in [s.LABEL, s.PRED, s.SEED]:
                        final[i] = tmp[i]
                    if origin_label_pred_sum == 0:
                        break

                # keep the seed/label/pred with largest target volume
                final_label_pred_sum = final[s.LABEL].sum() + final[s.PRED].sum()
                if tmp_label_pred_sum > final_label_pred_sum:
                    for i in [s.LABEL, s.PRED, s.SEED]:
                        final[i] = tmp[i]
                continue

            # target volume is large enough, break
            else:
                for i in [s.LABEL, s.PRED, s.SEED]:
                    final[i] = tmp[i]
                break

        # background
        background = 1 - final[s.LABEL]
        # !!! background FIRST !!!
        labels = torch.cat([background, final[s.LABEL]], dim=0)

        # load multi-modal imgs
        input_imgs = None
        multi_modal_list = [s.CT, s.PT, s.MR1, s.MR2]
        if self._no_pt:
            multi_modal_list.remove(s.PT)
        for i in multi_modal_list:
            img = self._preprocess(self.__origin[i], final[s.SEED])

            # concat multi-model img
            if input_imgs is None:
                input_imgs = img
            else:
                input_imgs = torch.cat([input_imgs, img], dim=0)

        # weight map
        weight_map = self._preprocess(
            img=self.__origin["weight.map"],
            augment_seed=final[s.SEED],
            normalize=False,
            clip_up_limit=self.__origin["weight.map"].max(),
        )

        return input_imgs, labels, weight_map


# # for testing
# weight = Dict()
# weight["fp.fn"] = 3
# weight[s.DISTANCE_STEP] = 10
# weight["slice"] = 2
# weight["prev.round.decay"] = 0.5
# weight[s.BACKGROUND] = 0.2

# augment = Dict()
# augment["methods"] = ["translate"]
# augment["pct"] = 1
# augment["min"] = 1
# augment["max"] = 1
# augment["times"] = 1

# selected_slices = Dict()
# selected_slices[s.ROUND_00] = [20, 40]
# selected_slices[s.ROUND_01] = [30]

# pred_dir = os.path.join(
#     g.TRAIN_RESULTS_DIR,
#     "baseline_2023.02.06.20.59.26_loss.delta=0.5_loss.gamma=0.3_optimal",
#     s.BASELINE,
#     "fold=01",
#     "epoch=171",
#     s.PATIENTS,
#     "patient=336",
# )
# # augment_methods = [translate,elastic,rotate,scale,flip.lr,flip.ud]
# tmp_dataset = DataSetIDLGTVt(
#     patient="336",
#     selected_slices=selected_slices,
#     label_dir=self._dataset_dir,
#     pred_dir=pred_dir,
#     augment=augment,
#     weight=weight,
# )
# tmp_dataset.__getitem__(2)
