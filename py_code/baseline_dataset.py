import global_elems as g
import os
import random
import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from torchvision import transforms as T
from data_augment import DataAugment


class BaselineDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_list: list,
        augment_pct: float = 0.0,
        augment_method: str = None,
        augment_low_limit: int = 1,
        augment_up_limit: int = 1,
    ):
        self.patient_list = patient_list
        self._init_augment(
            augment_pct=augment_pct,
            augment_method=augment_method,
            augment_low_limit=augment_low_limit,
            augment_up_limit=augment_up_limit,
        )
        # patient_slice_mapping is a list of: ["patient", "slice"]
        self._init_patient_slice_mapping(self.patient_list)

    def _init_augment(
        self,
        augment_pct: float = 0.0,
        augment_method: str = None,
        augment_low_limit: int = 1,
        augment_up_limit: int = 1,
    ):
        if augment_pct > 0.0 and augment_method is not None:
            self._data_augment = DataAugment(
                pct=augment_pct,
                method=augment_method,
                low_limit=augment_low_limit,
                up_limit=augment_up_limit,
            )
        else:
            self._data_augment = None

    def _init_patient_slice_mapping(self, patient_list: list):
        self.patient_slice_mapping = []
        for cur_patient in patient_list:
            cur_patient_folder = os.path.join(g.DATASET_FOLDER, cur_patient)

            # DO NOT shuffle here, shuffle when creating dataloader
            cur_patient_slices = g.get_sub_folders(cur_patient_folder, shuffle=False)

            for cur_slice in cur_patient_slices:
                self.patient_slice_mapping.append([cur_patient, cur_slice])

    # must be overrided
    def __len__(self):
        return len(self.patient_slice_mapping)

    def __pad_and_crop(
        self, input_img: ndarray, output_width: int, output_height: int
    ) -> ndarray:
        # step 1: padding
        input_height, input_width = input_img.shape
        pad_left = 0
        pad_right = 0
        pad_up = 0
        pad_down = 0
        if output_width > input_width:
            pad = output_width - input_width
            if pad % 2 == 0:
                pad_left = int(pad / 2)
                pad_right = pad_left
            else:
                pad_left = int(pad // 2)
                pad_right = pad_left + 1
        if output_height > input_height:
            pad = output_height - input_height
            if pad % 2 == 0:
                pad_up = int(pad / 2)
                pad_down = pad_up
            else:
                pad_up = int(pad // 2)
                pad_down = pad_up + 1
        if output_width > input_width or output_height > input_height:
            input_img = np.pad(
                input_img,
                ((pad_up, pad_down), (pad_left, pad_right)),
                "constant",
                constant_values=0,  # constant_values=0 means black padding
            )

        # step 2: cropping
        input_height, input_width = input_img.shape
        start_width = (input_width // 2) - (output_width // 2)
        start_height = (input_height // 2) - (output_height // 2)
        output_img = input_img[
            start_height : start_height + output_height,
            start_width : start_width + output_width,
        ]
        return output_img

    def __load_img(self, img_path: str, augment_seed: int):
        origin_img = np.load(img_path).astype(np.float32)

        origin_img = self.__pad_and_crop(
            input_img=origin_img,
            output_width=g.IMG_SIZE,
            output_height=g.IMG_SIZE,
        )
        if self._data_augment is not None:
            augment_img = self._data_augment.run(
                input_data=origin_img, seed=augment_seed
            )
        else:
            augment_img = origin_img

        # numpy to tensor
        to_tensor = T.ToTensor()
        return to_tensor(augment_img)

    def _get_item(
        self, cur_slice_folder: str, label_path: str
    ) -> tuple[Tensor, Tensor]:

        # make sure same group use the same augment seed
        # !!! use python random, do not use np.random !!!
        # np.random + dataloader will cause multi-processing problem
        augment_seed = random.randint(0, 2 ** 16)

        multi_model_img = None

        for i in ["ct", "pet", "mrt1", "mrt2"]:
            img_path = os.path.join(cur_slice_folder, i + ".npy")
            img = self.__load_img(img_path=img_path, augment_seed=augment_seed)
            # g.show_img(img)
            # print(img.max(), img.min())

            # concat multi-model img
            if multi_model_img is None:
                multi_model_img = img
            else:
                multi_model_img = torch.cat([multi_model_img, img], dim=0)

        # load label
        label_img = self.__load_img(img_path=label_path, augment_seed=augment_seed)
        # g.show_img(label_img, print_info=True)
        # print(label_img.max(), label_img.min())

        return multi_model_img, label_img

    # must be overrided
    def __getitem__(self, idx: int):
        cur_patient = self.patient_slice_mapping[idx][0]
        cur_slice = self.patient_slice_mapping[idx][1]
        cur_slice_folder = os.path.join(g.DATASET_FOLDER, cur_patient, cur_slice)
        label_path = os.path.join(cur_slice_folder, "label.npy")
        return self._get_item(cur_slice_folder=cur_slice_folder, label_path=label_path)


# for testing
# augment_method= translate / elastic / rotate / scale / combine
# if 0:
#     tmp_dataset = BaselineDataSet(
#         patient_list=["336"],
#         augment_method="combine",
#         augment_pct=1.0,
#         augment_low_limit=2,
#         augment_up_limit=2,
#     )
#     tmp_dataset.__getitem__(35)
