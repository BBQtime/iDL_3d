import os
import random
from typing import Tuple

import torch
from custom import Dict, Img, Nii
from dataset_core import DatasetCore
from torch import Tensor


class DataSetBaseline(DatasetCore):
    def __init__(
        self,
        patients: list,
        dataset_ver: str,
        no_pt: bool,
        augment: Dict = None,
    ):
        super().__init__(dataset_ver=dataset_ver, no_pt=no_pt, augment=augment)
        self.__patients = patients

    # must be overrided
    def __len__(self):
        return len(self.__patients)

    def get_item(self, patient: str) -> Tuple[Tensor, Tensor]:
        # load origin labels
        origin = Img.load_labels(dataset_dir=self._dataset_dir, patient=patient)
        tmp = Dict()
        final = Dict()

        # loop until target volume is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp["seed"] = random.randint(0, 2**16)

            # load gtvs
            tmp["gtvs"] = self._preprocess(origin["gtvs"], tmp["seed"])
            tmp["gtvs"] = Img.binarize(tmp["gtvs"])

            # target volume is not big enough
            if tmp["gtvs"].sum() < origin["gtvs"].sum() * 0.999:
                # keep the largest gtvs and the augment seed
                if final["gtvs"] == {} or tmp["gtvs"].sum() > final["gtvs"].sum():
                    final["gtvs"] = tmp["gtvs"]
                    final["seed"] = tmp["seed"]
                continue
            # target volume is large enough, break
            else:
                final["gtvs"] = tmp["gtvs"]
                final["seed"] = tmp["seed"]
                break

        # preprocess gtvt and gtvn based on final augment seed
        for gtv in ["gtvt", "gtvn"]:
            final[gtv] = self._preprocess(origin[gtv], final["seed"])
            final[gtv] = Img.binarize(final[gtv])

        # load background
        background = 1 - torch.maximum(final["gtvt"], final["gtvn"])
        # !!! background FIRST !!!
        labels = torch.cat([background, final["gtvt"], final["gtvn"]], dim=0)

        # load multi-modal imgs
        multi_modal_list = ["CT", "PT", "T1dr", "T2dr"]
        if self._no_pt:
            multi_modal_list.remove("PT")

        input_imgs = None
        for i in multi_modal_list:
            img_path = os.path.join(
                self._dataset_dir, "HNCDL_{}_{}.nii".format(patient, i)
            )
            img = Nii.load(img_path)

            # ct windowing before normalization
            if i == "CT":
                img = Img.ct_windowing(img)

            img = self._preprocess(img, final["seed"])

            # concat multi-model img
            if input_imgs is None:
                input_imgs = img
            else:
                input_imgs = torch.cat([input_imgs, img], dim=0)

        return input_imgs, labels

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)
