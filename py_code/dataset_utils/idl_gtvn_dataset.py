import os
import random

# import GeodisTK
import global_utils.global_core as g
import numpy as np
import torch
from dataset_utils.dataset_core import DatasetCore
from global_utils.custom_dict import Dict
from global_utils.str_lib import DatasetVer, ErrMsg, Modal
from numpy import ndarray
from scipy.ndimage import distance_transform_edt, measurements


class IDLGTVnDataSet(DatasetCore):
    def __init__(
        self,
        patients: list,
        dataset_ver: str,
        no_pt: bool,
        no_mr: bool,
        geodesic_distance: bool,
        augment: Dict = None,
        obs_gtvn_clicks: ndarray = None,
        random_click: bool = False,
    ):
        super().__init__(
            dataset_ver=dataset_ver,
            no_pt=no_pt,
            no_mr=no_mr,
            augment=augment,
        )
        self.__patients = patients
        self.__geodesic_distance = geodesic_distance
        self.__obs_gtvn_clicks = obs_gtvn_clicks
        self.__random_click = random_click

    # must be overrided
    def __len__(self):
        return len(self.__patients)

    def __load_gtvn_clicks(
        self,
        dataset_ver: str,
        patient: str,
        img_shape: tuple,  # if no gtvn/gtvn_clicks, create an empty img of this shape
    ):
        dataset_dir = g.DATASET_DIR[dataset_ver]

        if dataset_ver in [DatasetVer.AU, DatasetVer.OBS_STUDY]:
            gtvn_clicks_path = os.path.join(
                dataset_dir,
                "HNCDL_{}_GTVn_clicks.nii.gz".format(patient),
            )

        elif dataset_ver == DatasetVer.AU_EXT:
            gtvn_clicks_path = os.path.join(
                dataset_dir,
                "HNCDL_{}".format(patient),
                "HNCDL_{}_GTVn_clicks.nii.gz".format(patient),
            )

        elif dataset_ver == DatasetVer.NKI:
            gtvn_clicks_path = os.path.join(
                dataset_dir,
                patient,
                "{}_GTVn_clicks.nii.gz".format(patient),
            )

        elif dataset_ver == DatasetVer.HECKTOR:
            gtvn_clicks_path = os.path.join(
                dataset_dir,
                "{}_GTVn_clicks.nii.gz.gz".format(patient),
            )

        elif dataset_ver == DatasetVer.MDA:
            gtvn_clicks_path = os.path.join(
                dataset_dir,
                patient,
                "GTVn_clicks.nii.gz",
            )

        else:
            g.error_exit(ErrMsg.DATASET_VER_INVALID)

        if os.path.exists(gtvn_clicks_path):
            return g.load_nii(gtvn_clicks_path, binary=True)
        else:
            return np.zeros(img_shape, dtype=np.float32)

    # must be overrided
    def get_item(self, patient: str) -> Dict:

        # origin images dict
        self.__origin = Dict()

        # load label
        self.__origin["label"] = g.load_gtv_labels(
            dataset_ver=self._dataset_ver,
            patient=patient,
        )["gtvn"]

        # item to return
        item = Dict()
        # record img shape
        item["shape"] = self.__origin["label"].shape

        # find augment seed
        final = Dict()
        tmp = Dict()

        origin_label_sum = self.__origin["label"].sum()

        # loop until target volume is big enough
        for k in range(50):
            # make sure same group use the same augment_seed
            # !!! use python random, DO NOT use np.random !!!
            # np.random + dataloader will cause multi-processing problem
            tmp["augment.seed"] = random.randint(0, 2**16)

            # load gtvs
            tmp["label"] = self._preprocess(
                img=self.__origin["label"],
                augment_seed=tmp["augment.seed"],
            )
            tmp["label"] = g.binarize_img(tmp["label"])

            tmp_label_sum = tmp["label"].sum()

            # target volume is not large enough
            if tmp_label_sum < origin_label_sum * 0.999:
                # if "final" dict is empty
                if final == {}:
                    for i in ["label", "augment.seed"]:
                        final[i] = tmp[i]
                    if origin_label_sum == 0:
                        break

                # keep the seed/label/pred with largest target volume
                if tmp_label_sum > final["label"].sum():
                    for i in ["label", "augment.seed"]:
                        final[i] = tmp[i]
                continue

            # target volume is large enough, break
            else:
                for i in ["label", "augment.seed"]:
                    final[i] = tmp[i]
                break

        # background
        background = 1 - final["label"]
        # !!! background FIRST !!!
        item["labels"] = torch.cat([background, final["label"]], dim=0)

        # load multi modal imgs
        multi_modal_imgs = self._load_multi_modal_imgs(
            dataset_ver=self._dataset_ver,
            patient=patient,
            no_pt=self._no_pt,
            no_mr=self._no_mr,
        )
        for i in multi_modal_imgs.keys():
            self.__origin[i] = multi_modal_imgs[i]

        # (1) gtvn_clicks - observer study
        if self.__obs_gtvn_clicks is not None:
            self.__origin["clicks"] = self.__obs_gtvn_clicks

        # (2) gtvn_clicks - simulation
        else:
            self.__origin["clicks"] = self.__load_gtvn_clicks(
                dataset_ver=self._dataset_ver,
                patient=patient,
                img_shape=item["shape"],
            )
            # self.__origin["clicks"] = np.zeros(
            #     self.__origin["label"].shape, dtype=np.float32
            # )
            # # loop through each connected components
            # # cc_count = 1
            # for cur_gtvn_cc in g.get_connected_components(self.__origin["label"]):
            #     if self.__random_click:
            #         # random point (d,h,w)
            #         pos = g.get_random_nonzero_pos(cur_gtvn_cc)
            #     else:
            #         # gravity center: (d,h,w)
            #         pos = list(measurements.center_of_mass(cur_gtvn_cc))
            #         # float to int
            #         for i in range(len(pos)):
            #             pos[i] = round(pos[i])
            #     self.__origin["clicks"][pos[0]][pos[1]][pos[2]] = 1

        if 0:
            g.save_nii(
                self.__origin["clicks"],
                os.path.join(g.DEBUG_DIR, "gtvn_clicks.nii.gz"),
            )

        # generate distance map based on clicks
        if np.sum(self.__origin["label"]) > 0:

            # (1) geodesic distance map
            # if self.__geodesic_distance:
            if 0:
                # Get 3D geodesic disntance by raser scanning.
                # I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
                # Type should be np.float32.
                # S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
                # Type should be np.uint8.
                # spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
                # lamb: weighting betwween 0.0 and 1.0
                #     if lamb==0.0, return spatial euclidean distance without considering gradient
                #     if lamb==1.0, the distance is based on gradient only without using spatial distance
                # iter: number of iteration for raster scanning.

                # self.__origin["distance.map"] = GeodisTK.geodesic3d_raster_scan(
                #     g.normalize_img(self.__origin[Modal.CT]),
                #     self.__origin["clicks"].astype(np.uint8),
                #     (g.NII_SPACING[2], g.NII_SPACING[1], g.NII_SPACING[0]),
                #     1.0,  # lamb: weighting betwween 0.0 and 1.0
                #     4,  # iter: number of iteration for raster scanning.
                # )

                if 0:
                    g.save_nii(
                        self.__origin["distance.map"],
                        os.path.join(g.DEBUG_DIR, "geodesic_distance_map.nii.gz"),
                    )

            # (2) weighted Euclidean distance map
            else:
                self.__origin["distance.map"] = distance_transform_edt(
                    np.logical_not(self.__origin["clicks"])
                ).astype(np.float32)
                self.__origin["distance.map"] = np.exp(
                    -0.1 * self.__origin["distance.map"]
                )

                if 0:
                    g.save_nii(
                        self.__origin["distance.map"],
                        os.path.join(
                            g.DEBUG_DIR, "weighted_euclidean_distance_map.nii.gz"
                        ),
                    )

        else:
            self.__origin["distance.map"] = np.zeros_like(self.__origin["label"])

        item["input.imgs"] = None
        item["clicks"] = self._preprocess(
            img=self.__origin["clicks"],
            augment_seed=final["augment.seed"],
        )

        # pred + click
        for i in ["distance.map"]:
            final[i] = self._preprocess(
                img=self.__origin[i],
                augment_seed=final["augment.seed"],
            )
            if item["input.imgs"] is None:
                item["input.imgs"] = final[i]
            else:
                item["input.imgs"] = torch.cat([item["input.imgs"], final[i]], dim=0)

        # concatenate imput images
        for i in self.__origin.keys():
            if i in [Modal.CT, Modal.PT, Modal.MR1, Modal.MR2]:
                img = self._preprocess(
                    img=self.__origin[i],
                    augment_seed=final["augment.seed"],
                )

                # concat multi-model img
                item["input.imgs"] = torch.cat([item["input.imgs"], img], dim=0)

        # return item
        return item

    # must be overrided
    # this function is only for training, not for inference
    def __getitem__(self, idx: int):
        patient = self.__patients[idx]
        return self.get_item(patient)


# idl_gtvn_dataset = IDLGTVnDataSet(
#     patients=["106"],
#     dataset_ver=DatasetVer.AU,
#     no_pt=False,
#     no_mr=False,
#     geodesic_distance=False,
#     augment=None,
# )
# idl_gtvn_dataset.get_item("106")
