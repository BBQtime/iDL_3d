from custom import Global as g
import os
import random
import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from data_augment import DataAugmentation
from typing import Tuple
from custom import Dict
from custom import Nii
from custom import Img
from scipy.ndimage import measurements
from scipy.ndimage import distance_transform_edt


class IDLGTVnDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_list: list,
        pred_main_folder: str,
        augment: Dict = None,
    ):
        self.__patient_list = patient_list
        self.__pred_main_folder = pred_main_folder
        self.__augment = DataAugmentation(augment)

    # must be overrided
    def __len__(self):
        return len(self.__patient_list)

    def __preprocess(self, img: ndarray, augment_seed: int):
        # DO NOT alter origin img
        img = img.copy()

        # normalize before augmentation
        if not img.max() == img.min() == 0:
            img = Img.normalize(img)

        # data augmentation
        img = self.__augment.transform(input_data=img, seed=augment_seed)

        # no normalization after augmentation
        # because when rotating img
        # nomalization might give background a positive value

        # crop and pad after augmentation, max size: 89 283 280
        img = Img.central_pad(img, g.IMG_SHAPE)
        img = Img.central_crop(img, g.IMG_SHAPE)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, 1)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    # must be overrided
    def get_item(self, patient: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # origin images
        self.__origin = Dict()

        # load pred
        self.__origin["pred"] = Nii.load(
            os.path.join(
                self.__pred_main_folder, "patient={}".format(patient), "pred_gtvn.nii"
            ),
            binary=False,
        )

        # load label
        gtvn_path = os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVn.nii".format(patient))
        if os.path.exists(gtvn_path):
            self.__origin["label"] = Nii.load(
                gtvn_path,
                binary=True,
            )
        else:
            self.__origin["label"] = np.zeros_like(self.__origin["pred"])

        # load ct/pt/mrt1/mt2
        self.__origin["ct"] = Nii.load(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_CT.nii".format(patient))
        )
        # ct windowing before normalization
        self.__origin["ct"] = Img.ct_windowing(self.__origin["ct"])
        self.__origin["pt"] = Nii.load(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_PT.nii".format(patient))
        )
        self.__origin["mrt1"] = Nii.load(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_T1dr.nii".format(patient))
        )
        self.__origin["mrt2"] = Nii.load(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_T2dr.nii".format(patient))
        )

        # simulate click annotation
        self.__origin["click"] = np.zeros(
            self.__origin["label"].shape, dtype=np.float32
        )
        # loop through each connected components
        for cur_gtvn_cc in Img.connected_components(self.__origin["label"]):
            # label_center: (d,h,w)
            label_center = list(measurements.center_of_mass(cur_gtvn_cc))
            # float to int
            for i in range(len(label_center)):
                label_center[i] = round(label_center[i])
            self.__origin["click"][label_center[0]][label_center[1]][
                label_center[2]
            ] = 1
        # Nii.save(
        #     self.__origin["click"], os.path.join(g.PROJ_PATH, "debug", "click.nii")
        # )
        # Nii.save(self.__origin["label"], os.path.join(g.PROJ_PATH, "debug", "gtvn.nii"))

        if np.sum(self.__origin["label"]) > 0:
            self.__origin["click"] = distance_transform_edt(
                np.logical_not(self.__origin["click"])
            ).astype(np.float32)
            # Nii.save(
            #     self.__origin["click"],
            #     os.path.join(g.PROJ_PATH, "debug", "distance_map.nii"),
            # )

            self.__origin["click"] = np.exp(-self.__origin["click"])
            # Nii.save(
            #     self.__origin["click"],
            #     os.path.join(g.PROJ_PATH, "debug", "exp_distance_map.nii"),
            # )

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
                tmp[i] = self.__preprocess(
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

        # baseline pred (no binarization)
        # print(torch.median(final["pred"]))
        final["pred"] = self.__preprocess(self.__origin["pred"], final["seed"])
        # print(torch.median(final["pred"]))

        # click
        # geodesic distance map + euclidean distance map
        final["click"] = self.__preprocess(self.__origin["click"], final["seed"])

        # # save final nii files for debugging
        # for i in ["label", "pred", "click"]:
        #     Nii.save(final[i], os.path.join(g.PROJ_PATH, "debug", i + ".nii"))

        # multi model imgs
        multi_model_imgs = torch.cat([final["pred"], final["click"]], dim=0)
        for i in ["ct", "pt", "mrt1", "mrt2"]:
            img = self.__preprocess(self.__origin[i], final["seed"])

            # concat multi-model img
            multi_model_imgs = torch.cat([multi_model_imgs, img], dim=0)

        return multi_model_imgs, labels

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patient_list[idx]
        return self.get_item(patient)


# # for testing
# augment = Dict()
# # [translate,elastic,rotate,scale,flip.lr,flip.ud]
# augment["methods"] = ["flip.lr"]
# augment["pct"] = 1
# augment["min"] = 1
# augment["max"] = 1
# augment["times"] = 1

# pred_folder = os.path.join(
#     g.TRAIN_RESULTS_FOLDER,
#     "baseline_2023.02.27.07.08.09_loss.delta=0.5_loss.gamma=0.5_optimal",
#     "fold=01",
#     "epoch=205",
#     "baseline",
#     "patients",
#     "patient=336",
# )
# # augment_methods =
# tmp_dataset = IDLGTVnDataSet(
#     patient="106",
#     pred_folder=pred_folder,
#     augment=augment,
# )
# tmp_dataset.__getitem__(2)
