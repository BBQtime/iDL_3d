import os
import random
from typing import Tuple

import numpy as np
import torch
from custom import Dict
from custom import Global as g
from custom import Img, Nii
from data_augment import DataAugmentation
from numpy import ndarray
from scipy.ndimage import binary_dilation, distance_transform_edt, measurements
from torch import Tensor


class DataSetIDLGTVn(torch.utils.data.Dataset):
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
        self.__patients = patients
        self.__baseline_id = baseline_id
        self.__img_shape = g.IMG_SHAPE[dataset_ver]
        self.__dataset_dir = g.DATASET_DIR[dataset_ver]
        self.__no_pt = no_pt
        self.__augment = DataAugmentation(augment)
        self.__gtvn_clicks = gtvn_clicks
        self.__random_click = random_click

    # must be overrided
    def __len__(self):
        return len(self.__patients)

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
        img = Img.central_pad_and_crop(img, self.__img_shape)

        # clip, because data augmentation will sometime make img >1 or <0
        img = np.clip(img, 0, 1)

        # unsqueeze img to 4 dim before convert to Tensor
        img = np.expand_dims(img, axis=0)
        # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
        img = torch.from_numpy(img)
        return img

    # must be overrided
    def get_item(self, patient: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # origin images dict
        self.__origin = Dict()

        # load pred
        self.__origin["pred"] = Nii.load(
            os.path.join(
                g.TRAIN_RESULTS_DIR,
                self.__baseline_id,
                "baseline",
                "patients",
                "patient={}".format(patient),
                "gtvn_pred.nii",
            ),
            binary=False,
        )

        # load label
        self.__origin["label"] = Img.load_labels(
            dataset_dir=self.__dataset_dir, patient=patient
        )["gtvn"]

        # find augment seed
        final = Dict()
        tmp = Dict()

        # origin_pred needs to be binarized (without changing original img)
        # otherwise origin_label_pred_sum is too high
        origin_label_pred_sum = (
            self.__origin["label"].sum() + Img.binarize(self.__origin["pred"]).sum()
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

        # gtvn_clicks
        if self.__gtvn_clicks is not None:
            self.__origin["clicks"] = self.__gtvn_clicks
        else:
            # simulate click
            self.__origin["clicks"] = np.zeros(
                self.__origin["label"].shape, dtype=np.float32
            )
            # loop through each connected components
            # cc_count = 1
            for cur_gtvn_cc in Img.connected_components(self.__origin["label"]):
                if self.__random_click:
                    # random point (d,h,w)
                    pos = Img.find_random_point(cur_gtvn_cc)
                else:
                    # gravity center: (d,h,w)
                    pos = list(measurements.center_of_mass(cur_gtvn_cc))
                    # float to int
                    for i in range(len(pos)):
                        pos[i] = round(pos[i])
                self.__origin["clicks"][pos[0]][pos[1]][pos[2]] = 1
            # dilation
            if 0:
                # Nii.save(
                #     self.__origin["clicks"],
                #     os.path.join(g.PROJ_DIR, "debug", "before_dilation.nii"),
                # )
                structure = np.ones((5, 5, 5), dtype=np.float32)
                self.__origin["clicks"] = binary_dilation(
                    self.__origin["clicks"], structure
                ).astype(np.float32)
                # Nii.save(
                #     self.__origin["clicks"],
                #     os.path.join(g.PROJ_DIR, "debug", "after_dilation.nii"),
                # )

        # # debug save img
        # cur_click_nii = np.zeros_like(self.__origin["clicks"])
        # cur_click_nii[pos[0]][pos[1]][pos[2]] = 1
        # Nii.save(
        #     cur_gtvn_cc,
        #     os.path.join(g.PROJ_DIR, "debug", "cur_cc_{}.nii".format(cc_count)),
        # )
        # Nii.save(
        #     cur_click_nii,
        #     os.path.join(g.PROJ_DIR, "debug", "cur_click_{}.nii".format(cc_count)),
        # )
        # print(cc_count)
        # cc_count += 1

        # generate distance map based on clicks
        if np.sum(self.__origin["label"]) > 0:
            self.__origin["distance.map"] = distance_transform_edt(
                np.logical_not(self.__origin["clicks"])
            ).astype(np.float32)
            self.__origin["distance.map"] = np.exp(-0.1 * self.__origin["distance.map"])
        else:
            self.__origin["distance.map"] = np.zeros_like(self.__origin["label"])

        input_imgs = None
        clicks = self.__preprocess(self.__origin["clicks"], final["seed"])

        # pred + click
        for i in ["distance.map"]:  # ["pred", "distance.map"]:
            final[i] = self.__preprocess(self.__origin[i], final["seed"])
            if input_imgs is None:
                input_imgs = final[i]
            else:
                input_imgs = torch.cat([input_imgs, final[i]], dim=0)

        # load multi-modal imgs
        multi_modal_list = ["CT", "PT", "T1dr", "T2dr"]
        if self.__no_pt:
            multi_modal_list.remove("PT")
        for i in multi_modal_list:
            img_path = os.path.join(
                self.__dataset_dir, "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = Nii.load(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = Img.ct_windowing(img)

            img = self.__preprocess(img, final["seed"])

            # concat multi-model img
            input_imgs = torch.cat([input_imgs, img], dim=0)

        # None is used as a placeholder to ensure consistent return value formats for each dataset
        return input_imgs, labels, clicks

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)


# # for testing
# augment = Dict()
# # [translate,elastic,rotate,scale,flip.lr,flip.ud]
# augment["methods"] = []
# augment["pct"] = 1
# augment["min"] = 1
# augment["max"] = 1
# augment["times"] = 1

# baseline_epoch_dir = os.path.join(
#     g.TRAIN_RESULTS_DIR,
#     "baseline_2023.02.27.07.08.09_loss.gamma=0.5",
#     "fold=1",
#     "epoch=205",
# )
# # augment_methods =
# tmp_dataset = DataSetIDLGTVn(
#     patients=["129"],
#     baseline_epoch_dir=baseline_epoch_dir,
#     augment=None,
#     random_click=True,
# )
# tmp_dataset.__getitem__(0)
