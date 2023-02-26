import global_elems as g
import os
import torch
import numpy as np
from torch import Tensor
from typing import Tuple
import random
from data_augment import DataAugmentation
from numpy import ndarray
from nested_dict import NestedDict
from scipy.ndimage import distance_transform_edt

DEBUG_SAVE_IMG = False


class IDLDataSet:
    def __init__(
        self,
        patient: str,
        annotated_slices: dict,
        label_folder: str,
        pred_folder: str,
        augment: dict,
        weight: dict,
    ):
        self.patient = patient  # make this public

        if len(augment) == 0:
            self.__augment = DataAugmentation()
        else:
            self.__augment = DataAugmentation(
                methods=augment["methods"],
                pct=augment["pct"],
                low_limit=augment["low.limit"],
                up_limit=augment["up.limit"],
            )
        self.__augment_times = augment["times"]

        # save origin images
        self.__origin = NestedDict()

        # load label
        # (1) simulated iDL
        if label_folder == g.DATASET_FOLDER:
            # gtvs / gtvt
            self.__origin["label.gtvt"] = g.load_nii(
                os.path.join(label_folder, "HNCDL_{}_GTVt.nii".format(patient)),
                binary=True,
            )
            # gtvn
            label_gtvn_path = os.path.join(
                label_folder, "HNCDL_{}_GTVn.nii".format(patient)
            )
            if os.path.exists(label_gtvn_path):
                self.__origin["label.gtvn"] = g.load_nii(label_gtvn_path, binary=True)
            else:
                label_gtvs = g.load_nii(
                    os.path.join(label_folder, "HNCDL_{}_GTVs.nii".format(patient)),
                    binary=True,
                )
                self.__origin["label.gtvn"] = label_gtvs - self.__origin["label.gtvt"]
        # (2) real iDL
        else:
            for gtv in ["t", "n"]:
                self.__origin["label.gtv{}".format(gtv)] = g.load_nii(
                    os.path.join(label_folder, "pred_gtv{}".format(gtv)), binary=True
                )

        # load pred
        for gtv in ["t", "n"]:
            self.__origin["pred.gtv{}".format(gtv)] = g.load_nii(
                os.path.join(pred_folder, "pred_gtv{}.nii".format(gtv)), binary=True
            )

        # load ct/pt/mr1/mt2
        self.__origin["ct"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_CT.nii".format(patient))
        )
        # ct windowing before normalization
        self.__origin["ct"] = g.ct_windowing(self.__origin["ct"])
        self.__origin["pt"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_PT.nii".format(patient))
        )
        self.__origin["mrt1"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_T1dr.nii".format(patient))
        )
        self.__origin["mrt2"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_T2dr.nii".format(patient))
        )

        # weight map
        self.__origin["weight.map"], slice_mask = self.__load_weight_map(
            annotated_slices, weight
        )

        # overwrite pred to label on non-annotated slices
        if DEBUG_SAVE_IMG:
            g.save_nii(
                self.__origin["label.gtvt"],
                os.path.join(g.PROJ_PATH, "debug", "origin_label.nii"),
            )
            g.save_nii(
                self.__origin["pred.gtvt"],
                os.path.join(g.PROJ_PATH, "debug", "pred.nii"),
            )

        for gtv in ["t", "n"]:
            self.__origin["label.gtv{}".format(gtv)] *= slice_mask
            self.__origin["label.gtv{}".format(gtv)] += self.__origin[
                "pred.gtv{}".format(gtv)
            ] * (1 - slice_mask)

        if DEBUG_SAVE_IMG:
            g.save_nii(
                self.__origin["label.gtvt"],
                os.path.join(g.PROJ_PATH, "debug", "overwrite_label.nii"),
            )

    def __load_weight_map(self, annotated_slices: dict, weight: dict) -> dict:
        # annotated slice mask
        slice_mask = NestedDict()
        max_round = max(
            len(annotated_slices["transverse"]),
            len(annotated_slices["coronal"]),
            len(annotated_slices["sagittal"]),
        )

        for plane in ["transverse", "coronal", "sagittal"]:
            # annotated slice mask
            slice_mask[plane] = np.zeros(self.__origin["pt"].shape, dtype=np.float32)

            for cur_round in annotated_slices[plane]:
                # dont change weight["annotate.slice"], use another variable
                slice_weight = weight["annotated.slice"]
                cur_round_int = int(cur_round[len("round=") :])
                slice_weight *= pow(
                    weight["prev.round.decay"], (max_round - cur_round_int)
                )

                if slice_weight < weight["background"]:
                    slice_weight = weight["background"]

                # current step
                for cur_slice in annotated_slices[plane][cur_round]:
                    if plane == "transverse":
                        slice_mask[plane][cur_slice, :, :] = (
                            np.ones_like(slice_mask[plane][0, :, :]) * slice_weight
                        )
                    elif plane == "coronal":
                        slice_mask[plane][:, cur_slice, :] = (
                            np.ones_like(slice_mask[plane][:, 0, :]) * slice_weight
                        )
                    elif plane == "sagittal":
                        slice_mask[plane][:, :, cur_slice] = (
                            np.ones_like(slice_mask[plane][:, :, 0]) * slice_weight
                        )

        # get max value of each plane
        slice_mask = np.maximum(
            np.maximum(slice_mask["transverse"], slice_mask["coronal"]),
            slice_mask["sagittal"],
        )
        if DEBUG_SAVE_IMG:
            g.save_nii(
                slice_mask,
                os.path.join(g.PROJ_PATH, "debug", "slice_mask.nii"),
            )

        # get annotation (keep weight=1 before distance map)
        label_gtvs = np.maximum(
            self.__origin["label.gtvt"], self.__origin["label.gtvn"]
        )
        pred_gtvs = np.maximum(self.__origin["pred.gtvt"], self.__origin["pred.gtvn"])
        fp = pred_gtvs * (1 - label_gtvs)
        fn = (1 - pred_gtvs) * label_gtvs
        annotation = (fp + fn) * np.where(slice_mask > 0, 1, 0)
        annotation = annotation.astype(np.float32)

        # distance map
        distance_map = distance_transform_edt(np.logical_not(annotation))
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
        if DEBUG_SAVE_IMG:
            g.save_nii(
                distance_map, os.path.join(g.PROJ_PATH, "debug", "distance_map.nii")
            )

        # weighted annotation
        annotation = (
            annotation * slice_mask * (weight["annotation"] / weight["annotated.slice"])
        )
        if DEBUG_SAVE_IMG:
            g.save_nii(annotation, os.path.join(g.PROJ_PATH, "debug", "annotation.nii"))

        # final_weight_map
        weight_map = np.maximum(np.maximum(distance_map, slice_mask), annotation)
        if DEBUG_SAVE_IMG:
            g.save_nii(weight_map, os.path.join(g.PROJ_PATH, "debug", "weight_map.nii"))

        # return slice_mask to overwrite pred to label on non-annotated slices
        slice_mask = np.where(slice_mask > 0, 1, 0)

        return weight_map, slice_mask

    # must be overrided
    def __len__(self):
        return self.__augment_times

    def __preprocess(self, img: ndarray, augment_seed: int):
        # normalize before augmentation
        if not img.max() == img.min() == 0:
            img = g.normalize_img(img)

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # crop and pad after augmentation, max size: 89 283 280
        img = g.central_crop(img, g.IMG_SIZE)
        img = g.central_pad(img, g.IMG_SIZE)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    # must be overrided
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        final = NestedDict()
        tmp = NestedDict()

        origin_label_pred_sum = (
            self.__origin["label.gtvt"].sum()
            + self.__origin["label.gtvn"].sum()
            + self.__origin["pred.gtvt"].sum()
            + self.__origin["pred.gtvn"].sum()
        )

        # loop until target volume in patch is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            augment_seed = random.randint(0, 2**16)

            # load gtvs
            for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn"]:
                tmp[i] = self.__preprocess(self.__origin[i], augment_seed)

            tmp_label_pred_sum = (
                tmp["label.gtvt"].sum()
                + tmp["label.gtvn"].sum()
                + tmp["pred.gtvt"].sum()
                + tmp["pred.gtvn"].sum()
            )

            # target volume is not big enough
            if tmp_label_pred_sum < origin_label_pred_sum * 0.999:

                if len(final) == 0:
                    for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn"]:
                        final[i] = tmp[i]

                # keep the gtvt/gtvn with largest target volume
                final_label_pred_sum = (
                    final["label.gtvt"].sum()
                    + final["label.gtvn"].sum()
                    + final["pred.gtvt"].sum()
                    + final["pred.gtvn"].sum()
                )
                if tmp_label_pred_sum > final_label_pred_sum:
                    for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn"]:
                        final[i] = tmp[i]
                continue

            # target volume is large enough, break
            else:
                for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn"]:
                    final[i] = tmp[i]
                break

        # background
        background = 1 - final["label.gtvt"] - final["label.gtvn"]
        # !!! background FIRST !!!
        labels = torch.cat(
            [background, final["label.gtvt"], final["label.gtvn"]], dim=0
        )

        # multi model imgs
        multi_model_imgs = None
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            img = self.__preprocess(self.__origin[i], augment_seed)

            # concat multi-model img
            if multi_model_imgs is None:
                multi_model_imgs = img
            else:
                multi_model_imgs = torch.cat([multi_model_imgs, img], dim=0)

        # weight map
        weight_map = self.__preprocess(self.__origin["weight.map"], augment_seed)

        return multi_model_imgs, labels, weight_map


# # for testing
# if 0:
#     weight = dict()
#     weight["annotation"] = 3
#     weight["distance.step"] = 10
#     weight["annotated.slice"] = 2
#     weight["prev.round.decay"] = 0.5
#     weight["background"] = 0.2

#     augment = dict()
#     augment["methods"] = ["translate"]
#     augment["pct"] = 1
#     augment["low.limit"] = 1
#     augment["up.limit"] = 1
#     augment["times"] = 1

#     annotated_slices = dict()
#     annotated_slices["round=00"] = [20, 40]
#     annotated_slices["round=01"] = [30]

#     pred_folder = os.path.join(
#         g.TRAIN_RESULTS_FOLDER,
#         "baseline_2023.02.06.20.59.26_loss.delta=0.5_loss.gamma=0.3_optimal",
#         "baseline",
#         "fold=01",
#         "epoch=171",
#         "patients",
#         "patient=336",
#     )
#     # augment_methods = [translate,elastic,rotate,scale,flip.lr,flip.ud]
#     tmp_dataset = IDLDataSet(
#         patient="336",
#         annotated_slices=annotated_slices,
#         label_folder=g.DATASET_FOLDER,
#         pred_folder=pred_folder,
#         augment=augment,
#         weight=weight,
#     )
#     tmp_dataset.__getitem__(2)
