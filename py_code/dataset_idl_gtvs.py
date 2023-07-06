# from custom import Global as g
# import os
# import random
# import torch
# import numpy as np
# from numpy import ndarray
# from torch import Tensor
# from data_augment import DataAugmentation
# from typing import Tuple
# from custom import Dict
# from custom import Nii
# from custom import Img
# from scipy.ndimage import measurements
# from scipy.ndimage import distance_transform_edt
# from scipy.ndimage import binary_dilation


# class DataSetIDLGTVs(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         patients: list,
#         baseline_epoch_dir: str,
#         weight: Dict = None,
#         augment: Dict = None,
#         random_click: bool = False,
#     ):
#         self.__patients = patients
#         self.__baseline_epoch_dir = baseline_epoch_dir
#         self.__augment = DataAugmentation(augment)
#         self.__random_click = random_click
#         self.__weight = weight

#     # must be overrided
#     def __len__(self):
#         return len(self.__patients)

#     def __preprocess(
#         self,
#         img: ndarray,
#         augment_seed: int,
#         normalize: bool = True,  # =False for weight map
#         clip_up_limit: float = 1,  # !=1 for weight map
#     ):
#         # DO NOT alter origin img
#         img = img.copy()

#         # normalize before augmentation
#         if normalize and (not img.max() == img.min() == 0):
#             img = Img.normalize(img)

#         # data augmentation
#         img = self.__augment.transform(input_data=img, seed=augment_seed)

#         # no normalization after augmentation
#         # because when rotating img
#         # nomalization might give background a positive value

#         # crop and pad after augmentation, max size: 89 283 280
#         img = Img.central_resize(img, self.__img_shape)

#         # clip, because data augmentation will sometime make img >1 or <0
#         img = np.clip(img, 0, clip_up_limit)

#         # unsqueeze img to 4 dim before convert to Tensor
#         img = np.expand_dims(img, axis=0)
#         # do NOT use "T.ToTensor()" in 3D, it will make (d,h,w) to (h,d,w)
#         img = torch.from_numpy(img)
#         return img

#     def __load_gtvt_weight_map(self, label: ndarray, pred: ndarray, weight: Dict):
#         selected_slices = Dict()
#         # label_center: (d,h,w)
#         label_center = measurements.center_of_mass(label)
#         selected_slices["transverse"]["round=01"] = [round(label_center[0])]
#         selected_slices["coronal"]["round=01"] = [round(label_center[1])]
#         selected_slices["sagittal"]["round=01"] = [round(label_center[2])]

#         # annotated slice mask
#         slice_mask = Dict()
#         max_round = max(
#             len(selected_slices["transverse"]),
#             len(selected_slices["coronal"]),
#             len(selected_slices["sagittal"]),
#         )

#         for plane in ["transverse", "coronal", "sagittal"]:
#             # annotated slice mask
#             slice_mask[plane] = np.zeros(label.shape, dtype=np.float32)

#             for round_num in selected_slices[plane]:
#                 # do NOT change weight["annotate.slice"], use another variable
#                 slice_weight = weight["slice"]
#                 slice_weight *= pow(
#                     weight["prev.round.decay"],
#                     (max_round - int(round_num[len("round=") :])),
#                 )
#                 if slice_weight < weight["background"]:
#                     slice_weight = weight["background"]

#                 # current step
#                 for slice_num in selected_slices[plane][round_num]:
#                     if plane == "transverse":
#                         slice_mask[plane][slice_num, :, :] = (
#                             np.ones_like(slice_mask[plane][0, :, :]) * slice_weight
#                         )
#                     elif plane == "coronal":
#                         slice_mask[plane][:, slice_num, :] = (
#                             np.ones_like(slice_mask[plane][:, 0, :]) * slice_weight
#                         )
#                     elif plane == "sagittal":
#                         slice_mask[plane][:, :, slice_num] = (
#                             np.ones_like(slice_mask[plane][:, :, 0]) * slice_weight
#                         )

#         # combine slice_mask on 3 planes
#         slice_mask = np.maximum(
#             np.maximum(slice_mask["transverse"], slice_mask["coronal"]),
#             slice_mask["sagittal"],
#         )
#         # Nii.save(slice_mask, os.path.join(g.PROJ_DIR, "debug", "slice_mask.nii"))

#         # get fp&fn (keep weight=1 before creating distance map)
#         fp = pred * (1 - label)
#         fn = (1 - pred) * label
#         fp_fn = fp + fn
#         fp_fn = fp_fn * np.where(slice_mask > 0, 1, 0)
#         fp_fn = fp_fn.astype(np.float32)

#         # annotation (pred + label)
#         if 1:
#             annotation = np.maximum(pred, label)
#             annotation = annotation * np.where(slice_mask > 0, 1, 0)
#             annotation = annotation.astype(np.float32)
#             # Nii.save(annotation, os.path.join(g.PROJ_DIR, "debug", "annotation.nii"))

#         # distance map
#         if 1:
#             distance_map = distance_transform_edt(np.logical_not(annotation))
#         else:
#             distance_map = distance_transform_edt(np.logical_not(fp_fn))
#         distance_map = distance_map.astype(np.float32)
#         distance_map = np.where(
#             distance_map >= 2 * weight["distance.step"],
#             -weight["background"],
#             distance_map,
#         )
#         distance_map = np.where(
#             distance_map >= weight["distance.step"],
#             -weight["background"] / 2,
#             distance_map,
#         )
#         distance_map = np.where(distance_map >= 0, 0, distance_map)
#         distance_map *= -1
#         # Nii.save(distance_map, os.path.join(g.PROJ_DIR, "debug", "distance_map.nii"))

#         # weighted fp&fn (after weight map)
#         fp_fn = fp_fn * slice_mask * (weight["fp.fn"] / weight["slice"])
#         # Nii.save(fp_fn, os.path.join(g.PROJ_DIR, "debug", "fp_fn.nii"))

#         # final_weight_map
#         weight_map = np.maximum(np.maximum(distance_map, slice_mask), fp_fn)
#         # Nii.save(weight_map, os.path.join(g.PROJ_DIR, "debug", "weight_map.nii"))

#         # return slice_mask to overwrite pred to label on non-annotated slices
#         slice_mask = np.where(slice_mask > 0, 1, 0)

#         annotation = (label * slice_mask).astype(np.float32)

#         return weight_map, annotation

#     # load distance map based on simulated clicks
#     def __load_distance_map(self, label):
#         clicks = np.zeros(label.shape, dtype=np.float32)

#         # loop through each connected components
#         for cur_gtvn_cc in Img.connected_components(label):
#             if self.__random_click:
#                 # random point (d,h,w)
#                 pos = Img.find_random_point(cur_gtvn_cc)
#             else:
#                 # gravity center: (d,h,w)
#                 pos = list(measurements.center_of_mass(cur_gtvn_cc))
#                 # float to int
#                 for i in range(len(pos)):
#                     pos[i] = round(pos[i])
#             clicks[pos[0]][pos[1]][pos[2]] = 1

#         # dilation
#         if 0:
#             structure = np.ones((5, 5, 5), dtype=np.float32)
#             clicks = binary_dilation(clicks, structure).astype(np.float32)

#         # generate distance map based on clicks
#         if np.sum(label) > 0:
#             distance_map = distance_transform_edt(np.logical_not(clicks)).astype(
#                 np.float32
#             )
#             distance_map = np.exp(-0.1 * distance_map)
#         else:
#             distance_map = np.zeros_like(label)

#         return distance_map, clicks

#     # must be overrided
#     def get_item(self, patient: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#         # origin images dict
#         origin = Dict()

#         # load preds
#         for gtv in ["t", "n"]:
#             origin["pred.gtv{}".format(gtv)] = Nii.load(
#                 os.path.join(
#                     self.__baseline_epoch_dir,
#                     "baseline",
#                     "patients",
#                     "patient={}".format(patient),
#                     "pred_gtv{}.nii".format(gtv),
#                 ),
#                 binary=False,
#             )

#         # load labels
#         for gtv in ["t", "n"]:
#             origin["label.gtv{}".format(gtv)] = Nii.load(
#                 os.path.join(g.DATASET_DIR, "HNCDL_{}_GTV{}.nii".format(patient, gtv)),
#                 binary=True,
#             )

#         # find augment seed
#         final = Dict()
#         tmp = Dict()

#         # origin_pred needs to be binarized (without changing original img),
#         # otherwise origin_label_pred_sum is too high
#         origin_label_pred_sum = (
#             origin["label.gtvt"].sum()
#             + origin["label.gtvn"].sum()
#             + Img.binarize(origin["pred.gtvt"]).sum()
#             + Img.binarize(origin["pred.gtvn"]).sum()
#         )

#         # loop until target volume is big enough
#         for k in range(50):
#             # make sure same group use the same augment_seed
#             # !!! use python random, DO NOT use np.random !!!
#             # np.random + dataloader will cause multi-processing problem
#             tmp["seed"] = random.randint(0, 2**16)

#             # load gtvs
#             tmp_label_pred_sum = 0
#             for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn"]:
#                 tmp[i] = self.__preprocess(img=origin[i], augment_seed=tmp["seed"])
#                 tmp[i] = Img.binarize(tmp[i])
#                 tmp_label_pred_sum += tmp[i].sum()

#             # target volume is not large enough
#             if tmp_label_pred_sum < origin_label_pred_sum * 0.999:

#                 # if "final" dict is empty
#                 if final == {}:
#                     for i in [
#                         "label.gtvt",
#                         "label.gtvn",
#                         "pred.gtvt",
#                         "pred.gtvn",
#                         "seed",
#                     ]:
#                         final[i] = tmp[i]
#                     if origin_label_pred_sum == 0:
#                         break

#                 # keep the seed/label/pred with largest target volume
#                 final_label_pred_sum = 0
#                 for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn"]:
#                     final_label_pred_sum += final[i].sum()
#                 if tmp_label_pred_sum > final_label_pred_sum:
#                     for i in [
#                         "label.gtvt",
#                         "label.gtvn",
#                         "pred.gtvt",
#                         "pred.gtvn",
#                         "seed",
#                     ]:
#                         final[i] = tmp[i]
#                 continue

#             # target volume is large enough, break
#             else:
#                 for i in ["label.gtvt", "label.gtvn", "pred.gtvt", "pred.gtvn", "seed"]:
#                     final[i] = tmp[i]
#                 break

#         # background
#         background = 1 - torch.maximum(final["label.gtvt"], final["label.gtvn"])
#         # !!! background FIRST !!!
#         labels = torch.cat(
#             [background, final["label.gtvt"], final["label.gtvn"]], dim=0
#         )

#         # load weight map
#         origin_gtvt_weight_map, origin_gtvt_annotation = self.__load_gtvt_weight_map(
#             label=origin["label.gtvt"],
#             pred=origin["pred.gtvt"],
#             weight=self.__weight,
#         )

#         (
#             origin_gtvn_distance_map,
#             origin_gtvn_clicks,
#         ) = self.__load_distance_map(origin["label.gtvn"])

#         final_gtvn_clicks = self.__preprocess(origin_gtvn_clicks, final["seed"])

#         # weight map
#         final_gtvt_weight_map = self.__preprocess(
#             img=origin_gtvt_weight_map,
#             augment_seed=final["seed"],
#             normalize=False,
#             clip_up_limit=origin_gtvt_weight_map.max(),
#         )

#         # gtvt.annotation + gtvn.distance.map
#         input_imgs = self.__preprocess(origin_gtvt_annotation, final["seed"])
#         input_imgs = torch.cat(
#             [input_imgs, self.__preprocess(origin_gtvn_distance_map, final["seed"])],
#             dim=0,
#         )

#         # load ct/pt/mr1/mr2
#         for i in ["CT", "PT", "T1dr", "T2dr"]:
#             img_path = os.path.join(g.DATASET_DIR, "HNCDL_{}_{}.nii".format(patient, i))
#             img = Nii.load(img_path)

#             # ct windowing before normalization
#             if i == "CT":
#                 img = Img.ct_windowing(img)

#             img = self.__preprocess(img, final["seed"])

#             # concat multi-model img
#             input_imgs = torch.cat([input_imgs, img], dim=0)

#         # None is used as a placeholder to ensure consistent return value formats for each dataset
#         return input_imgs, labels, final_gtvt_weight_map, final_gtvn_clicks

#     # must be overrided
#     # this function is only for training, not for inference
#     def __getitem__(self, idx: int):
#         patient = self.__patients[idx]
#         return self.get_item(patient)


# # # for testing
# # augment = Dict()
# # # [translate,elastic,rotate,scale,flip.lr,flip.ud]
# # augment["methods"] = []
# # augment["pct"] = 1
# # augment["min"] = 1
# # augment["max"] = 1
# # augment["times"] = 1

# # baseline_epoch_dir = os.path.join(
# #     g.TRAIN_RESULTS_DIR,
# #     "baseline_2023.02.27.07.08.09_loss.gamma=0.5",
# #     "fold=1",
# #     "epoch=205",
# # )
# # # augment_methods =
# # tmp_dataset = DataSetIDLGTVn(
# #     patients=["129"],
# #     baseline_epoch_dir=baseline_epoch_dir,
# #     augment=None,
# #     random_click=True,
# # )
# # tmp_dataset.__getitem__(0)
