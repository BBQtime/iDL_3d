import os
import random
from typing import Tuple

import torch
from custom import Dict, Img, Nii
from dataset_core import DatasetCore
from str_lib import StrLib as s
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
            tmp[s.SEED] = random.randint(0, 2**16)

            # load gtvs
            tmp[s.GTVS] = self._preprocess(origin[s.GTVS], tmp[s.SEED])
            tmp[s.GTVS] = Img.binarize(tmp[s.GTVS])

            # target volume is not big enough
            if tmp[s.GTVS].sum() < origin[s.GTVS].sum() * 0.999:
                # keep the largest gtvs and the augment seed
                if final[s.GTVS] == {} or tmp[s.GTVS].sum() > final[s.GTVS].sum():
                    final[s.GTVS] = tmp[s.GTVS]
                    final[s.SEED] = tmp[s.SEED]
                continue
            # target volume is large enough, break
            else:
                final[s.GTVS] = tmp[s.GTVS]
                final[s.SEED] = tmp[s.SEED]
                break

        # preprocess gtvt and gtvn based on final augment seed
        for gtv in [s.GTVT, s.GTVN]:
            final[gtv] = self._preprocess(origin[gtv], final[s.SEED])
            final[gtv] = Img.binarize(final[gtv])

        # load background
        background = 1 - torch.maximum(final[s.GTVT], final[s.GTVN])
        # !!! background FIRST !!!
        labels = torch.cat([background, final[s.GTVT], final[s.GTVN]], dim=0)

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

            img = self._preprocess(img, final[s.SEED])

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


# for testing
# augment_methods=[translate / elastic / rotate / scale / flip.lr / flip.ud]
# patients without GTVn: 257 192
if 0:
    augment = Dict()
    # [translate,elastic,rotate,scale,flip.lr,flip.ud]
    augment["methods"] = []
    augment["pct"] = 1
    augment["min"] = 1
    augment["max"] = 1
    augment["times"] = 1
    tmp_dataset = DataSetBaseline(
        patients=["257"],
        augment=augment,
    )
    tmp_dataset.__getitem__(0)
