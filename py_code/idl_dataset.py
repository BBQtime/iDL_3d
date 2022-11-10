import global_elems as g
import os
import math
from baseline_dataset import BaselineDataSet
from torch import Tensor
from typing import Tuple


class IDLDataSet(BaselineDataSet):
    def __init__(
        self,
        patient: str,
        slice_dict: dict,
        label_folder: str = g.DATASET_FOLDER,
        augment_times: int = 1,
        augment_pct: float = 0.0,
        augment_method: str = None,
        augment_low_limit: int = 1,
        augment_up_limit: int = 1,
    ):
        self.patient = patient
        self.__label_folder = label_folder
        self._init_augment(
            augment_pct=augment_pct,
            augment_method=augment_method,
            augment_low_limit=augment_low_limit,
            augment_up_limit=augment_up_limit,
        )

        # patient_slice_mapping is a list of: ["patient", "slice"]
        self._init_patient_slice_mapping(self.patient, slice_dict, augment_times)

    def _init_patient_slice_mapping(
        self, patient: str, slice_dict: dict, augment_times: int
    ):
        self.patient_slice_mapping = []
        for cur_round in reversed(slice_dict):
            # current step
            for cur_slice in slice_dict[cur_round]:
                for i in range(augment_times):
                    self.patient_slice_mapping.append([patient, cur_slice])
            if augment_times >= 16:
                augment_times /= 4
            else:
                augment_times /= 2
            # rounded up
            augment_times = math.ceil(augment_times)

    # must be overrided
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        cur_patient = self.patient_slice_mapping[idx][0]
        cur_slice = self.patient_slice_mapping[idx][1]
        cur_slice_folder = os.path.join(g.DATASET_FOLDER, cur_patient, cur_slice)

        # (1) use label in dataset folder
        if self.__label_folder == g.DATASET_FOLDER:
            label_path = os.path.join(cur_slice_folder, "label.npy")

        # (2) post processing, load filtered slices from "post.process" folder
        # (3) real training, load slices annotated by the doctor, not from dataset
        else:
            # change slice id format from "0XX" to "XX"
            if cur_slice.startswith("0"):
                cur_slice = cur_slice[1:]

            label_path = os.path.join(
                self.__label_folder, ("slice_" + cur_slice + "_label.npy")
            )

        return self._get_item(
            cur_slice_folder=cur_slice_folder,
            label_path=label_path,
        )


# for testing
# augment_method = translate / elastic / rotate / scale / combine
# if 0:
#     slice_dict = NestedDict()
#     slice_dict["round=00"] = ["035", "042"]
#     slice_dict["round=01"] = ["024"]
#     tmp_dataset = IDLDataset(
#         patient="336",
#         slice_dict=slice_dict,
#         label_folder=None,
#         augment_times=8,
#         augment_method="combine",
#         augment_pct=1.0,
#         augment_low_limit=2,
#         augment_up_limit=2,
#     )
#     tmp_dataset.__getitem__(1)
