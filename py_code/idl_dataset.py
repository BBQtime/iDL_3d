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
        augment_methods: list = [],
        augment_times: int = 1,
        augment_pct: float = 0,
        augment_low_limit: int = 0,
        augment_up_limit: int = 0,
    ):
        self.patient = patient  # make this public

        self.__augment = DataAugmentation(
            pct=augment_pct,
            methods=augment_methods,
            low_limit=augment_low_limit,
            up_limit=augment_up_limit,
        )
        self.__augment_times = augment_times

        self.__origin = NestedDict()

        # load label
        # (1) simulated iDL
        if label_folder == g.DATASET_FOLDER:
            self.__origin["label.gtvt"] = g.load_nii(
                os.path.join(label_folder, "HNCDL_{}_GTVt.nii".format(patient)),
                binary=True,
            )
            label_gtvn_path = os.path.join(
                label_folder, "HNCDL_{}_GTVn.nii".format(patient)
            )
            if os.path.exists(label_gtvn_path):
                self.__origin["label.gtvn"] = g.load_nii(label_gtvn_path, binary=True)
            else:
                label_gtvs_img = g.load_nii(
                    os.path.join(label_folder, "HNCDL_{}_GTVs.nii".format(patient)),
                    binary=True,
                )
                self.__origin["label.gtvn"] = (
                    label_gtvs_img - self.__origin["label.gtvt"]
                )

        # (2) real iDL
        else:
            self.__origin["label.gtvt"] = g.load_nii(
                os.path.join(label_folder, "pred_gtvt.nii"), binary=True
            )
            self.__origin["label.gtvn"] = g.load_nii(
                os.path.join(label_folder, "pred_gtvn.nii"), binary=True
            )

        # load pred
        self.__origin["pred.gtvt"] = g.load_nii(
            os.path.join(pred_folder, "pred_gtvt.nii"), binary=True
        )
        self.__origin["pred.gtvn"] = g.load_nii(
            os.path.join(pred_folder, "pred_gtvn.nii"), binary=True
        )

        # load ct/pt/mr1/mt2
        self.__origin["ct"] = g.ct_preprocess(
            g.load_nii(
                os.path.join(g.DATASET_FOLDER, "HNCDL_{}_CT.nii".format(patient))
            )
        )
        self.__origin["pt"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_PT.nii".format(patient))
        )
        self.__origin["mrt1"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_T1dr.nii".format(patient))
        )
        self.__origin["mrt2"] = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_T2dr.nii".format(patient))
        )

        self.__origin["weight.map"] = self.__load_weight_map(annotated_slices)

        # for cur_round in reversed(annotated_slices):
        #     # current step
        #     for cur_slice in annotated_slices[cur_round]:
        #         # augmentation times
        #         for times in range(augment_times):
        #             self.__annotated_slices[idx]["slice.id"] = cur_slice
        #             self.__annotated_slices[idx]["patch.pos"] = tuple(patch_pos)
        #     if augment_times >= 16:
        #         augment_times /= 4
        #     else:
        #         augment_times /= 2
        #     # rounded up
        #     augment_times = math.ceil(augment_times)

    def __load_weight_map(self, annotated_slices):
        annotated_slices = g.dict_to_list(annotated_slices)
        annotated_slices = g.list_remove_duplicates(annotated_slices)

        weight_map = np.zeros(self.__origin["pt"].shape, dtype=np.float32)

        for i in range(weight_map.shape[0]):
            if i in annotated_slices:
                weight_map[i] = np.ones_like(weight_map[i])

        g.save_nii(weight_map, os.path.join(g.PROJ_PATH, "debug", "weight_map.nii"))
        return weight_map

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

        # weight_map
        weight_map = self.__preprocess(self.__origin["weight.map"], augment_seed)

        return multi_model_imgs, labels, weight_map


# for testing
if 0:
    annotated_slices = dict()
    annotated_slices["round=00"] = [20, 40]
    annotated_slices["round=01"] = [30]
    pred_folder = os.path.join(
        g.TRAIN_RESULTS_FOLDER,
        "baseline_2023.02.06.20.59.26_loss.delta=0.5_loss.gamma=0.3_optimal",
        "baseline",
        "fold=01",
        "epoch=171",
        "patients",
        "patient=336",
    )
    # augment_methods = [translate,elastic,rotate,scale,flip.lr,flip.ud]
    tmp_dataset = IDLDataSet(
        patient="336",
        annotated_slices=annotated_slices,
        label_folder=g.DATASET_FOLDER,
        pred_folder=pred_folder,
        augment_times=1,
        augment_methods=["translate"],
        augment_pct=1,
        augment_low_limit=1,
        augment_up_limit=1,
    )
    tmp_dataset.__getitem__(2)
