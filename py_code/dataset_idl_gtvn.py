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
from scipy.ndimage import binary_dilation, distance_transform_edt, measurements
from str_lib import CLICKS, DISTANCE_MAP, GTVN, LABEL, PRED, SEED
from torch import Tensor


class DataSetIDLGTVn(DatasetCore):
    def __init__(
        self,
        patients: list,
        baseline_id: str,
        dataset_ver: str,
        no_pt: bool,
        augment: Dict = None,
        gtvn_clicks: ndarray = None,
        random_click: bool = False,
    ):
        super().__init__(dataset_ver=dataset_ver, no_pt=no_pt, augment=augment)
        self.__patients = patients
        self.__baseline_id = baseline_id
        self.__gtvn_clicks = gtvn_clicks
        self.__random_click = random_click

    # must be overrided
    def __len__(self):
        return len(self.__patients)

    # must be overrided
    def get_item(self, patient: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # origin images dict
        self.__origin = Dict()

        # load pred
        self.__origin[PRED] = Nii.load(
            os.path.join(
                g.TRAIN_RESULTS_DIR,
                self.__baseline_id,
                "baseline",
                "patients",
                "patient={}".format(patient),
                "gtvn_pred.nii.gz",
            ),
            binary=False,
        )

        # load label
        self.__origin[LABEL] = Img.load_labels(
            dataset_dir=self._dataset_dir, patient=patient
        )[GTVN]

        # find augment seed
        final = Dict()
        tmp = Dict()

        # origin_pred needs to be binarized (without changing original img)
        # otherwise origin_label_pred_sum is too high
        origin_label_pred_sum = (
            self.__origin[LABEL].sum() + Img.binarize(self.__origin[PRED]).sum()
        )

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

        # gtvn_clicks
        if self.__gtvn_clicks is not None:
            self.__origin[CLICKS] = self.__gtvn_clicks
        else:
            # simulate click
            self.__origin[CLICKS] = np.zeros(
                self.__origin[LABEL].shape, dtype=np.float32
            )
            # loop through each connected components
            # cc_count = 1
            for cur_gtvn_cc in Img.connected_components(self.__origin[LABEL]):
                if self.__random_click:
                    # random point (d,h,w)
                    pos = Img.find_random_point(cur_gtvn_cc)
                else:
                    # gravity center: (d,h,w)
                    pos = list(measurements.center_of_mass(cur_gtvn_cc))
                    # float to int
                    for i in range(len(pos)):
                        pos[i] = round(pos[i])
                self.__origin[CLICKS][pos[0]][pos[1]][pos[2]] = 1

        # generate distance map based on clicks
        if np.sum(self.__origin[LABEL]) > 0:
            self.__origin[DISTANCE_MAP] = distance_transform_edt(
                np.logical_not(self.__origin[CLICKS])
            ).astype(np.float32)
            self.__origin[DISTANCE_MAP] = np.exp(-0.1 * self.__origin[DISTANCE_MAP])
        else:
            self.__origin[DISTANCE_MAP] = np.zeros_like(self.__origin[LABEL])

        input_imgs = None
        clicks = self._preprocess(self.__origin[CLICKS], final[SEED])

        # pred + click
        for i in [DISTANCE_MAP]:  # [PRED, DISTANCE_MAP]:
            final[i] = self._preprocess(self.__origin[i], final[SEED])
            if input_imgs is None:
                input_imgs = final[i]
            else:
                input_imgs = torch.cat([input_imgs, final[i]], dim=0)

        # load multi-modal imgs
        multi_modal_list = ["CT", "PT", "T1dr", "T2dr"]
        if self._no_pt:
            multi_modal_list.remove("PT")
        for i in multi_modal_list:
            img_path = os.path.join(
                self._dataset_dir, "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = Nii.load(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = Img.ct_windowing(img)

            img = self._preprocess(img, final[SEED])

            # concat multi-model img
            input_imgs = torch.cat([input_imgs, img], dim=0)

        # None is used as a placeholder to ensure consistent return value formats for each dataset
        return input_imgs, labels, clicks

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)
