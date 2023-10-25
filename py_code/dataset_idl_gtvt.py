import os
import random
from typing import Tuple

import numpy as np
import torch
from custom import Dict, Img, Modal, Nii, Plane
from dataset_core import DatasetCore
from numpy import ndarray
from scipy.ndimage import distance_transform_edt
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
            label_path = os.path.join(label_dir, "gtvt_label.nii.gz")

        # load label
        self.__origin["label"] = Nii.load(label_path, binary=True)

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
        self.__origin["weight.map"], slice_mask = self.__load_weight_map(
            selected_slices, weight
        )

        # overwrite pred to label on non-annotated slices
        self.__origin["label"] *= slice_mask
        self.__origin["label"] += self.__origin["pred"] * (1 - slice_mask)

    def __load_weight_map(self, selected_slices: Dict, weight: Dict):
        # annotated slice mask
        slice_mask = Dict()
        max_round = max(
            len(selected_slices[Plane.TRANSVERSE]),
            len(selected_slices[Plane.CORONAL]),
            len(selected_slices[Plane.SAGITTAL]),
        )

        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            # annotated slice mask
            slice_mask[plane] = np.zeros(
                self.__origin[Modal.CT].shape, dtype=np.float32
            )

            for round_num in selected_slices[plane]:
                # do NOT change weight["annotate.slice"], use another variable
                slice_weight = weight["slice"]
                slice_weight *= pow(
                    weight["prev.round.decay"],
                    (max_round - int(round_num[len("round=") :])),
                )
                if slice_weight < weight["background"]:
                    slice_weight = weight["background"]

                # current step
                for slice_num in selected_slices[plane][round_num]:
                    if plane == Plane.TRANSVERSE:
                        slice_mask[plane][slice_num, :, :] = (
                            np.ones_like(slice_mask[plane][0, :, :]) * slice_weight
                        )
                    elif plane == Plane.CORONAL:
                        slice_mask[plane][:, slice_num, :] = (
                            np.ones_like(slice_mask[plane][:, 0, :]) * slice_weight
                        )
                    elif plane == Plane.SAGITTAL:
                        slice_mask[plane][:, :, slice_num] = (
                            np.ones_like(slice_mask[plane][:, :, 0]) * slice_weight
                        )

        # combine slice_mask on 3 planes
        slice_mask = np.maximum(
            np.maximum(slice_mask[Plane.TRANSVERSE], slice_mask[Plane.CORONAL]),
            slice_mask[Plane.SAGITTAL],
        )

        # get fp&fn (keep weight=1 before creating distance map)
        fp = self.__origin["pred"] * (1 - self.__origin["label"])
        fn = (1 - self.__origin["pred"]) * self.__origin["label"]
        fp_plus_fn = fp + fn
        fp_plus_fn = fp_plus_fn * np.where(slice_mask > 0, 1, 0)
        fp_plus_fn = fp_plus_fn.astype(np.float32)

        # pred union label
        pred_union_label = np.maximum(self.__origin["pred"], self.__origin["label"])
        pred_union_label = pred_union_label * np.where(slice_mask > 0, 1, 0)
        pred_union_label = pred_union_label.astype(np.float32)

        # distance map
        if 1:
            distance_map = distance_transform_edt(np.logical_not(pred_union_label))
        else:
            distance_map = distance_transform_edt(np.logical_not(fp_plus_fn))
        distance_map = distance_map.astype(np.float32)
        distance_map = np.where(
            distance_map >= 2 * weight["distance.step"],
            -weight["background"],
            distance_map,
        )
        distance_map = np.where(
            distance_map >= weight["distance.step"],
            -weight["background"] / 2,
            distance_map,
        )
        distance_map = np.where(distance_map >= 0, 0, distance_map)
        distance_map *= -1

        # weighted fp&fn (after weight map)
        fp_plus_fn = fp_plus_fn * slice_mask * (weight["fp.fn"] / weight["slice"])

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


# # for testing
# weight = Dict()
# weight["fp.fn"] = 3
# weight["distance.step"] = 10
# weight["slice"] = 2
# weight["prev.round.decay"] = 0.5
# weight["background"] = 0.2

# augment = Dict()
# augment["methods"] = ["translate"]
# augment["pct"] = 1
# augment["min"] = 1
# augment["max"] = 1
# augment["times"] = 1

# selected_slices = Dict()
# selected_slices["round=00"] = [20, 40]
# selected_slices["round=01"] = [30]

# pred_dir = os.path.join(
#     g.TRAIN_RESULTS_DIR,
#     "baseline_2023.02.06.20.59.26_loss.delta=0.5_loss.gamma=0.3_optimal",
#     "baseline",
#     "fold=01",
#     "epoch=171",
#     "patients",
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
