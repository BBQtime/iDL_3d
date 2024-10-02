import hashlib

import global_utils.global_core as g
import torch
from dataset_utils.dataset_core import DatasetCore
from global_utils.custom_dict import Dict


class BaselineDataSet(DatasetCore):
    def __init__(
        self,
        patients: list,
        dataset_ver: str,
        no_pt: bool,
        no_mr: bool,
        augment: Dict = None,
    ):
        super().__init__(
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            no_mr=no_mr,
            augment=augment,
        )
        self.__patients = patients
        self.current_epoch = 0

    def _generate_seed(self, patient_id) -> int:
        """Generate a deterministic seed based on the patient ID and epoch number."""
        combined_id = f"{patient_id}_{self.current_epoch}"
        return int(hashlib.sha256(combined_id.encode("utf-8")).hexdigest(), 16) % 2**16

    def set_epoch(self, epoch: int):
        """Sets the current epoch to be used for seed generation."""
        self.current_epoch = epoch

    # must be overrided
    def __len__(self):
        return len(self.__patients)

    def get_item(self, patient: str) -> Dict:

        # load labels and save them into "origin" dict
        origin = g.load_gtv_labels(
            dataset_ver=self._dataset_ver,
            patient=patient,
        )

        # data to return
        item = Dict()
        # record img shape
        item["shape"] = origin["gtvt"].shape

        final = Dict()

        augment_seed = self._generate_seed(patient)
        final["augment.seed"] = augment_seed

        # preprocess gtvt and gtvn based on final augment seed
        for gtv in ["gtvt", "gtvn"]:
            final[gtv] = self._preprocess(
                img=origin[gtv],
                augment_seed=final["augment.seed"],
            )
            final[gtv] = g.binarize_img(final[gtv])

        # load background
        background = 1 - torch.maximum(final["gtvt"], final["gtvn"])
        # !!! background FIRST !!!
        item["labels"] = torch.cat([background, final["gtvt"], final["gtvn"]], dim=0)

        # load input imgs
        item["input.imgs"] = None
        multi_modal_imgs = self._load_multi_modal_imgs(
            dataset_ver=self._dataset_ver,
            patient=patient,
            no_pt=self._no_pt,
            no_mr=self._no_mr,
        )
        for i in multi_modal_imgs.keys():
            # preprocess (normalization, augmentation, center alignment, to tensor)
            multi_modal_imgs[i] = self._preprocess(
                img=multi_modal_imgs[i],
                augment_seed=final["augment.seed"],
            )

            # concat multi-model img
            if item["input.imgs"] is None:
                item["input.imgs"] = multi_modal_imgs[i]
            else:
                item["input.imgs"] = torch.cat(
                    [item["input.imgs"], multi_modal_imgs[i]], dim=0
                )

        return item

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)


# baseline_dataset = BaselineDataSet(
#     patients=["3259451405"],
#     dataset_ver=DatasetVer.MDA,
#     no_pt=True,
#     augment=None,
# )
# baseline_dataset.get_item("3259451405")
