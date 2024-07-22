import os
import random
from typing import Tuple

import global_core as g
import numpy as np
import torch
from custom_dict import Dict
from dataset_core import DatasetCore
from numpy import ndarray
from scipy.ndimage import distance_transform_edt, measurements
from torch import Tensor


class DataSetIDLGTVn(DatasetCore):
    def __init__(
        self,
        patients: list,
        baseline_id: str,
        dataset_ver: str,
        no_pt: bool,
        augment: Dict = None,
        obs_gtvn_clicks: ndarray = None,
        random_click: bool = False,
    ):
        super().__init__(dataset_ver=dataset_ver, no_pt=no_pt, augment=augment)
        self.__patients = patients
        self.__baseline_id = baseline_id
        self.__gtvn_clicks = obs_gtvn_clicks
        self.__random_click = random_click

    # must be overrided
    def __len__(self):
        return len(self.__patients)

    # must be overrided
    def get_item(
        self,
        patient: str,
        mda_obs: str = None,  # for MDA dataset, has multiple observers. None means random (for training)
    ) -> Dict:

        # origin images dict
        self.__origin = Dict()

        # load label
        self.__origin["label"] = g.load_gtv_labels(
            dataset_ver=self._dataset_ver,
            patient=patient,
            mda_obs=mda_obs,
        )["gtvn"]

        # no label found, return None
        if self.__origin["label"] is None:
            return None

        # item to return
        item = Dict()
        # record img shape
        item["shape"] = self.__origin["label"].shape

        # load pred
        self.__origin["pred"] = g.load_nii(
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

        # find augment seed
        final = Dict()
        tmp = Dict()

        # origin_pred needs to be binarized (without changing original img)
        # otherwise origin_label_pred_sum is too high
        origin_label_pred_sum = (
            self.__origin["label"].sum() + g.binarize_img(self.__origin["pred"]).sum()
        )

        # loop until target volume is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp["augment.seed"] = random.randint(0, 2**16)

            # load gtvs
            for i in ["label", "pred"]:
                tmp[i] = self._preprocess(
                    img=self.__origin[i],
                    augment_seed=tmp["augment.seed"],
                )
                tmp[i] = g.binarize_img(tmp[i])

            tmp_label_pred_sum = tmp["label"].sum() + tmp["pred"].sum()

            # target volume is not large enough
            if tmp_label_pred_sum < origin_label_pred_sum * 0.999:
                # if "final" dict is empty
                if final == {}:
                    for i in ["label", "pred", "augment.seed"]:
                        final[i] = tmp[i]
                    if origin_label_pred_sum == 0:
                        break

                # keep the seed/label/pred with largest target volume
                final_label_pred_sum = final["label"].sum() + final["pred"].sum()
                if tmp_label_pred_sum > final_label_pred_sum:
                    for i in ["label", "pred", "augment.seed"]:
                        final[i] = tmp[i]
                continue

            # target volume is large enough, break
            else:
                for i in ["label", "pred", "augment.seed"]:
                    final[i] = tmp[i]
                break

        # background
        background = 1 - final["label"]
        # !!! background FIRST !!!
        labels = torch.cat([background, final["label"]], dim=0)

        # gtvn_clicks
        # (1) observer study
        if self.__gtvn_clicks is not None:
            self.__origin["clicks"] = self.__gtvn_clicks
        # (2) simulation
        else:
            self.__origin["clicks"] = np.zeros(
                self.__origin["label"].shape, dtype=np.float32
            )
            # loop through each connected components
            # cc_count = 1
            for cur_gtvn_cc in g.get_connected_components(self.__origin["label"]):
                if self.__random_click:
                    # random point (d,h,w)
                    pos = g.get_random_nonzero_pos(cur_gtvn_cc)
                else:
                    # gravity center: (d,h,w)
                    pos = list(measurements.center_of_mass(cur_gtvn_cc))
                    # float to int
                    for i in range(len(pos)):
                        pos[i] = round(pos[i])
                self.__origin["clicks"][pos[0]][pos[1]][pos[2]] = 1

        # generate distance map based on clicks
        if np.sum(self.__origin["label"]) > 0:
            self.__origin["distance.map"] = distance_transform_edt(
                np.logical_not(self.__origin["clicks"])
            ).astype(np.float32)
            self.__origin["distance.map"] = np.exp(-0.1 * self.__origin["distance.map"])
        else:
            self.__origin["distance.map"] = np.zeros_like(self.__origin["label"])

        input_imgs = None
        clicks = self._preprocess(
            img=self.__origin["clicks"],
            augment_seed=final["augment.seed"],
        )

        # pred + click
        for i in ["distance.map"]:  # ["pred", "distance.map"]:
            final[i] = self._preprocess(
                img=self.__origin[i],
                augment_seed=final["augment.seed"],
            )
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
                g.DATASET_DIR[self._dataset_ver], "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = g.load_nii(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = g.windowing_ct(img)

            img = self._preprocess(
                img=img,
                augment_seed=final["augment.seed"],
            )

            # concat multi-model img
            input_imgs = torch.cat([input_imgs, img], dim=0)

        # return item
        item["input.imgs"] = input_imgs
        item["labels"] = labels
        item["clicks"] = clicks
        return item

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)
