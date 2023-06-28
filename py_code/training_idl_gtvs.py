# import os
# import torch
# import numpy as np
# from training_baseline import TrainingBaseline
# from dataset_idl_gtvs import DataSetIDLGTVs
# from custom import Folder
# from custom import Json
# from custom import Explorer
# from custom import Nii
# from custom import Dict
# from custom import List
# from custom import Img
# from custom import GPU
# from custom import ValueUtils
# from pathlib import Path
# from tqdm import tqdm
# from custom import Global as g
# from torch.nn import DataParallel
# from unet_pp import UNetPPSlim


# class TrainingIDLGTVs(TrainingBaseline):
#     def _load_common_hyper(
#         self,
#         hyper: Dict,
#         baseline_epoch_dir: str = None,  # this is only for idl.gtvn and idl
#         debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
#     ):
#         # epochs
#         if debug_mode:
#             # at least 2 epochs to compare loss difference
#             hyper["epochs"] = 2
#         else:
#             hyper["epochs"] = ValueUtils.limit_range(hyper["epochs"], (1, None))

#         # record actual epochs because of early stop
#         hyper["epochs.actual"] = 0

#         # early stop, based on epoch
#         hyper["early.stop.epochs"] = ValueUtils.limit_range(
#             hyper["early.stop.epochs"], (1, hyper["epochs"])
#         )

#         # lr
#         hyper["lr"] = ValueUtils.limit_range(hyper["lr"], (g.EPS, 1.0))

#         # actual lr
#         if GPU.used_count() > 1:
#             hyper["lr.actual"] = hyper["lr"] * GPU.used_count()
#         else:
#             hyper["lr.actual"] = hyper["lr"]

#         # min lr
#         hyper["lr.min"] = ValueUtils.limit_range(hyper["lr.min"], (g.EPS, hyper["lr"]))

#         # lr decay patience, based on epoch, must be defined before shared_hyper()
#         hyper["lr.decay.patience"] = ValueUtils.limit_range(
#             hyper["lr.decay.patience"], (1, hyper["epochs"])
#         )

#         # number of best valid loss cnn retained
#         hyper["keep.best.cnn.num"] = ValueUtils.limit_range(
#             hyper["keep.best.cnn.num"], (1, hyper["epochs"])
#         )

#         # augment percent
#         hyper["augment.pct"] = ValueUtils.limit_range(hyper["augment.pct"], (0.0, 1.0))

#         # load shared hyper parameters
#         super()._load_common_hyper(hyper=hyper, cnn_path=None)

#         # loss function
#         hyper["loss.func"] = UnifiedFocalLoss(
#             asym=hyper["loss.asym"],
#             weight=hyper["loss.weight"],
#             delta=hyper["loss.delta"],
#             gamma=hyper["loss.gamma"],
#             train_type=hyper["train.type"],
#         ).to(g.DEVICE)

#         # load patients
#         (train_patients, valid_patients, test_patients,) = self._load_patients(
#             fold=fold,
#             debug_mode=debug_mode,
#         )

#         # create datasets
#         # run this after shared hyper loaded, because hyper["augment"] is needed
#         augment = Dict()
#         augment["methods"] = hyper["augment.methods"]
#         augment["pct"] = hyper["augment.pct"]
#         augment["min"] = hyper["augment.min"]
#         augment["max"] = hyper["augment.max"]

#         # weight map parameters
#         if hyper["train.type"] == "idl":
#             hyper["weight.background"] = ValueUtils.limit_range(
#                 hyper["weight.background"], (0.0, 1.0)
#             )
#             hyper["weight.slice"] = ValueUtils.limit_range(
#                 hyper["weight.slice"], (hyper["weight.background"], None)
#             )
#             hyper["weight.fp.fn"] = ValueUtils.limit_range(
#                 hyper["weight.fp.fn"], (hyper["weight.slice"], None)
#             )
#             hyper["weight.distance.step"] = ValueUtils.limit_range(
#                 hyper["weight.distance.step"], (1, None)
#             )
#             hyper["weight.prev.round.decay"] = ValueUtils.limit_range(
#                 hyper["weight.prev.round.decay"], (0.0, 1.0)
#             )
#             weight = Dict()
#             weight["background"] = hyper["weight.background"]
#             weight["distance.step"] = hyper["weight.distance.step"]
#             weight["fp.fn"] = hyper["weight.fp.fn"]
#             weight["prev.round.decay"] = hyper["weight.prev.round.decay"]
#             weight["slice"] = hyper["weight.slice"]

#         # load train/valid/test datasets
#         if hyper["train.type"] == "baseline":
#             train_set = DataSetBaseline(patients=train_patients, augment=augment)
#             valid_set = DataSetBaseline(patients=valid_patients)
#             test_set = DataSetBaseline(patients=test_patients)

#         elif hyper["train.type"] == "idl_gtvn":
#             train_set = DataSetIDLGTVn(
#                 patients=train_patients,
#                 baseline_epoch_dir=baseline_epoch_dir,
#                 augment=augment,
#                 random_click=False,
#                 # random_click=True,
#             )
#             valid_set = DataSetIDLGTVn(
#                 patients=valid_patients,
#                 baseline_epoch_dir=baseline_epoch_dir,
#                 random_click=False,
#             )
#             test_set = DataSetIDLGTVn(
#                 patients=test_patients,
#                 baseline_epoch_dir=baseline_epoch_dir,
#                 random_click=False,
#             )

#         elif hyper["train.type"] == "idl":
#             train_set = DataSetIDLGTVs(
#                 patients=train_patients,
#                 baseline_epoch_dir=baseline_epoch_dir,
#                 weight=weight,
#                 augment=augment,
#                 random_click=False,
#             )
#             valid_set = DataSetIDLGTVs(
#                 patients=valid_patients,
#                 baseline_epoch_dir=baseline_epoch_dir,
#                 weight=weight,
#                 random_click=False,
#             )
#             test_set = DataSetIDLGTVs(
#                 patients=test_patients,
#                 baseline_epoch_dir=baseline_epoch_dir,
#                 weight=weight,
#                 random_click=False,
#             )

#         # dataloader
#         hyper["train.loader"] = DataLoader(
#             dataset=train_set,
#             batch_size=hyper["batch.size.actual"],
#             shuffle=True,  # only shuffle train loader
#             num_workers=g.NUM_WORKERS,
#         )
#         hyper["valid.loader"] = DataLoader(
#             dataset=valid_set,
#             batch_size=hyper["batch.size.actual"],
#             shuffle=False,
#             num_workers=g.NUM_WORKERS,
#         )
#         hyper["test.loader"] = DataLoader(
#             dataset=test_set,
#             batch_size=hyper["batch.size.actual"],
#             shuffle=False,
#             num_workers=g.NUM_WORKERS,
#         )

#     # if float64 needed, use: "cnn.to(torch.double)"
#     def __load_cnn(self, hyper: Dict, cnn_path: str = None):
#         # new model
#         if cnn_path == "" or cnn_path is None:
#             hyper["cnn"] = UNetPPSlim(
#                 in_chan=6, out_chan=3, dropout=hyper["dropout"]
#             ).to(g.DEVICE)

#         # existing model
#         else:
#             hyper["cnn"] = torch.load(cnn_path).to(g.DEVICE)

#         # set multi-GPU
#         if GPU.used_count() > 1:
#             hyper["cnn"] = DataParallel(hyper["cnn"]).to(g.DEVICE)

#     def _calculate_loss(self, item, hyper: Dict):
#         input_imgs = item[0].to(g.DEVICE)
#         labels = item[1].to(g.DEVICE)
#         weight_map = item[2].to(g.DEVICE)
#         preds = hyper["cnn"](input_imgs)
#         loss = hyper["loss.func"](preds, labels, weight_map)
#         return loss

#     def new_training(
#         self,
#         baseline_id: str = None,
#         baseline_fold: int = None,
#         baseline_epoch: int = None,
#         train_remark: str = "",
#         debug_mode: bool = False,
#     ):
#         if train_type == "baseline":
#             hyper_json_path = g.HYPER_JSON_PATH_BASELINE
#         elif train_type == "idl_gtvn":
#             hyper_json_path = g.HYPER_JSON_PATH_IDL_GTVN
#         else:
#             train_type = "idl"
#             hyper_json_path = g.HYPER_JSON_PATH_IDL_GTVS

#         for hyper in self._load_hyper_sets_from_json(hyper_json_path):

#             # add training type into hyper
#             hyper["train.type"] = train_type

#             train_id = hyper["train.type"] + "_"
#             train_id += self._init_train_id(
#                 train_remark=train_remark,
#                 hyper_json_path=hyper_json_path,
#                 hyper=hyper,
#                 debug_mode=debug_mode,
#             )
#             print("")
#             print(train_id)

#             # find baseline fold dir
#             if hyper["train.type"] == "baseline":
#                 baseline_epoch_dir = None
#             else:
#                 if baseline_fold is None or baseline_fold <= 0:
#                     key_word = "fold="
#                 else:
#                     key_word = "fold={}".format(baseline_fold)
#                 baseline_fold_dir = Explorer.get_sub_folders(
#                     os.path.join(g.TRAIN_RESULTS_DIR, baseline_id),
#                     key_word=key_word,
#                     full_path=True,
#                 )[0]
#                 # find epoch folder
#                 if baseline_epoch is None or baseline_epoch <= 0:
#                     key_word = "epoch="
#                 else:
#                     key_word = "epoch={:03d}".format(baseline_epoch)
#                 baseline_epoch_dir = Explorer.get_sub_folders(
#                     baseline_fold_dir, key_word=key_word, full_path=True
#                 )[0]

#             # create train result dir
#             if hyper["train.type"] == "baseline":
#                 train_result_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id)
#             else:
#                 train_result_dir = os.path.join(
#                     baseline_epoch_dir, hyper["train.type"], train_id
#                 )
#             Folder.create(train_result_dir)

#             # cross validation
#             hyper["cross.valid.fold"] = int(hyper["cross.valid.fold"])
#             hyper["cross.valid.fold"] = ValueUtils.limit_range(
#                 hyper["cross.valid.fold"], (0, g.DATASET_K_FOLDS)
#             )
#             if hyper["cross.valid.fold"] == 0:
#                 fold_list = List(range(1, g.DATASET_K_FOLDS + 1))
#             else:
#                 fold_list = [hyper["cross.valid.fold"]]

#             # loop through each fold
#             for fold in fold_list:
#                 fold_dir = os.path.join(train_result_dir, "fold={}".format(fold))
#                 Folder.create(fold_dir)

#                 # load and print hyperparams
#                 self._load_unique_hyper(
#                     hyper=hyper,
#                     fold=fold,
#                     baseline_epoch_dir=baseline_epoch_dir,
#                     debug_mode=debug_mode,
#                 )
#                 print("")
#                 self._print_hyper(hyper)

#                 print("")
#                 print("cross validation fold: {}".format(fold))

#                 train_info_dir = os.path.join(fold_dir, "train_info")
#                 Folder.create(train_info_dir)
#                 # save an empty loss.json
#                 Json.save(Dict(), os.path.join(train_info_dir, "loss.json"))
#                 # save an empty lr.json
#                 Json.save(Dict(), os.path.join(train_info_dir, "lr.json"))

#                 # save hyper before training
#                 hyper_save_path = os.path.join(train_info_dir, "hyper.json")
#                 self._save_hyper(hyper, hyper_save_path)

#                 # start training
#                 hyper["time.spent"] = datetime.now()
#                 self._training_traverse_epochs(hyper, fold_dir)
#                 hyper["time.spent"] = datetime.now() - hyper["time.spent"]
#                 hyper["time.spent"] = str(hyper["time.spent"]).split(".", 2)[0]

#                 # save hyper after training
#                 self._save_hyper(hyper, hyper_save_path)

#                 # clear time spent before next training
#                 hyper.pop("time.spent")

#                 # only train 2 folds in debug mode
#                 if debug_mode and fold_list.index(fold) == 1:
#                     break

#             # inference (valid set first)
#             for dataset in ["valid", "test.inter"]:
#                 self._inference(
#                     train_id=train_id,
#                     dataset=dataset,
#                     debug_mode=debug_mode,
#                 )

#     def __single_patient_inference(
#         self,
#         patient: str,
#         hyper: Dict,
#         baseline_epoch_dir: str = None,  # dataset needs this
#     ) -> Dict:
#         result = Dict()  # gtv->metric

#         origin = Dict()  # original labels

#         weight = Dict()
#         weight["background"] = 0.2
#         weight["distance.step"] = 2
#         weight["fp.fn"] = 1
#         weight["prev.round.decay"] = 0.5
#         weight["slice"] = 1
#         dataset = DataSetIDLGTVs(
#             patients=[patient],
#             baseline_epoch_dir=baseline_epoch_dir,
#             weight=weight,
#             random_click=False,
#         )

#         # load labels
#         for gtv in ["s", "t", "n"]:
#             origin["gtv{}".format(gtv)] = Nii.load(
#                 os.path.join(g.DATASET_DIR, "HNCDL_{}_GTV{}.nii".format(patient, gtv)),
#                 binary=True,
#             )

#         # get pred
#         input_imgs, labels, gtvt_weight_map, gtvn_clicks = dataset.get_item(
#             patient=patient
#         )
#         # add "batch" (c/d/h/w -> b/c/d/h/w)
#         input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)
#         labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
#         hyper["cnn"].eval()  # disable dropout / batch nomalize
#         with torch.no_grad():
#             preds = hyper["cnn"].forward(input_imgs)
#         # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
#         preds = torch.squeeze(preds, dim=0).cpu().numpy()

#         result["gtvt"]["pred"] = preds[1]
#         result["gtvn"]["pred"] = preds[2]
#         result["gtvs"]["pred"] = np.maximum(preds[1], preds[2])

#         # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
#         input_imgs = torch.squeeze(input_imgs, dim=0).cpu().numpy()
#         result["gtvt"]["annotation"] = input_imgs[0]
#         result["gtvn"]["distance.map"] = input_imgs[1]
#         # squeeze "channel" (c/d/h/w -> d/h/w)
#         result["gtvt"]["weight.map"] = (
#             torch.squeeze(gtvt_weight_map, dim=0).cpu().numpy()
#         )
#         result["gtvn"]["clicks"] = torch.squeeze(gtvn_clicks, dim=0).cpu().numpy()

#         # pad and crop to original size
#         # 1.preds
#         for gtv in ["gtvs", "gtvt", "gtvn"]:
#             result[gtv]["pred"] = Img.central_pad_and_crop(
#                 result[gtv]["pred"], origin["gtvs"].shape
#             )
#         # 2.annotation and weight_map
#         for i in ["annotation", "weight.map"]:
#             result["gtvt"][i] = Img.central_pad_and_crop(
#                 result["gtvt"][i], origin["gtvs"].shape
#             )
#         # 3.distance_map and clicks
#         for i in ["distance.map", "clicks"]:
#             result["gtvn"][i] = Img.central_pad_and_crop(
#                 result["gtvn"][i], origin["gtvs"].shape
#             )

#         # gtvt post processing
#         if 1:
#             cc_list = Img.connected_components(result["gtvt"]["pred"])
#             result["gtvt"]["pred"] = np.zeros_like(result["gtvt"]["pred"])
#             for cur_cc in cc_list:
#                 if (cur_cc * result["gtvt"]["annotation"]).sum() > 0:
#                     result["gtvt"]["pred"] = np.maximum(result["gtvt"]["pred"], cur_cc)

#         # idl_gtvn post processing
#         if 0:
#             cc_list = Img.connected_components(result["gtvn"]["pred"])
#             result["gtvn"]["pred"] = np.zeros_like(result["gtvn"]["pred"])
#             for cur_cc in cc_list:
#                 if (cur_cc * result["gtvn"]["clicks"]).sum() > 0:
#                     result["gtvn"]["pred"] = np.maximum(result["gtvn"]["pred"], cur_cc)

#         # calculate inference scores
#         for gtv in ["gtvs", "gtvt", "gtvn"]:
#             for metric in g.METRICS:
#                 result[gtv][metric] = self._metrics[metric](
#                     result[gtv]["pred"], origin[gtv]
#                 )
#         return result

#     def inference(self, idl_gtvs_id: str, debug_mode: bool = False):
#         print("")
#         print("inference: {}".format(idl_gtvs_id))

#         # find idl folder
#         idl_gtvs_dir = self._find_train_dir(idl_gtvs_id)
#         if idl_gtvs_dir is None:
#             print("idl_gtvs_id not found")
#             return

#         baseline_epoch_dir = str(Path(idl_gtvs_dir).parent.parent)

#         # loop through fold dirs
#         for fold_dir in Explorer.get_sub_folders(
#             idl_gtvs_dir, key_word="fold=", full_path=True
#         ):
#             fold = int(Path(fold_dir).name[len("fold=") :])
#             print("")
#             print("fold: ", fold)

#             # loop through epoch dirs
#             for epoch_dir in Explorer.get_sub_folders(
#                 fold_dir, key_word="epoch=", full_path=True
#             ):
#                 epoch = int(Path(epoch_dir).name[len("epoch=") :])
#                 print("epoch: ", epoch)

#                 # load cnn
#                 cnn_path = Explorer.get_sub_files(
#                     epoch_dir, key_word=".pt", full_path=True
#                 )[0]
#                 hyper = Dict()  # create an empty dict to save cnn
#                 hyper["dropout"] = 0
#                 self.__load_cnn(hyper=hyper, cnn_path=cnn_path)

#                 # load test patients
#                 patients = self._load_patients(fold=fold, debug_mode=debug_mode)["test.inter"]

#                 # initialize scores dict
#                 epoch_scores = Dict()
#                 gtv_list = ["gtvs", "gtvt", "gtvn"]

#                 # copy baseline scores
#                 baseline_scores = Json.load(
#                     os.path.join(baseline_epoch_dir, "baseline", "inference_test_inter.json")
#                 )
#                 key_list = ["median", "avg"]
#                 for patient in patients:
#                     key_list.append("patient={}".format(patient))
#                 for i in key_list:
#                     for gtv in ["gtvs", "gtvt", "gtvn"]:
#                         for metric in g.METRICS:
#                             epoch_scores[i][gtv][metric]["round=00"] = baseline_scores[
#                                 i
#                             ][gtv][metric]

#                 # initialize median and avg score of round=01
#                 for stats in ["median", "avg"]:
#                     for gtv in gtv_list:
#                         for metric in g.METRICS:
#                             epoch_scores[i][gtv][metric]["round=01"] = List()

#                 # loop through each patient
#                 for patient in tqdm(patients):
#                     # create folder to save cur patient preds and scores
#                     patient_dir = os.path.join(
#                         epoch_dir,
#                         "patients",
#                         "patient={}".format(patient),
#                     )
#                     Folder.create(patient_dir)

#                     # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
#                     patient_results = self.__single_patient_inference(
#                         patient=patient,
#                         hyper=hyper,
#                         baseline_epoch_dir=baseline_epoch_dir,
#                     )

#                     # save Niis of current patient
#                     Nii.save(
#                         img=patient_results["gtvt"]["annotation"],
#                         save_path=os.path.join(patient_dir, "gtvt_annotation.nii"),
#                         spacing=g.NII_SPACING,
#                     )
#                     Nii.save(
#                         img=patient_results["gtvt"]["weight.map"],
#                         save_path=os.path.join(patient_dir, "gtvt_weight_map.nii"),
#                         spacing=g.NII_SPACING,
#                     )
#                     Nii.save(
#                         img=patient_results["gtvn"]["distance.map"],
#                         save_path=os.path.join(patient_dir, "gtvn_distance_map.nii"),
#                         spacing=g.NII_SPACING,
#                     )
#                     Nii.save(
#                         img=patient_results["gtvn"]["clicks"],
#                         save_path=os.path.join(patient_dir, "gtvn_clicks.nii"),
#                         spacing=g.NII_SPACING,
#                     )
#                     for gtv in gtv_list:
#                         Nii.save(
#                             img=patient_results[gtv]["pred"],
#                             save_path=os.path.join(patient_dir, "{}_pred.nii".format(gtv)),
#                             spacing=g.NII_SPACING,
#                         )

#                     # record score of current patient
#                     for gtv in gtv_list:
#                         for metric in g.METRICS:
#                             # save cur patient score into inference_test.json (test set only)
#                             epoch_scores["patient={}".format(patient)][gtv][metric][
#                                 "round=01"
#                             ] = patient_results[gtv][metric]
#                             # add scores of current patient into median(list)
#                             for stats in ["median", "avg"]:
#                                 epoch_scores[i][gtv][metric]["round=01"].append(
#                                     patient_results[gtv][metric]
#                                 )

#                 # all patients under current epoch have been traversed
#                 # calculate avg and median score
#                 for gtv in gtv_list:
#                     for metric in g.METRICS:
#                         epoch_scores["median"][gtv][metric][
#                             "round=01"
#                         ] = ValueUtils.median(
#                             epoch_scores["median"][gtv][metric]["round=01"]
#                         )
#                         epoch_scores["avg"][gtv][metric]["round=01"] = ValueUtils.avg(
#                             epoch_scores["avg"][gtv][metric]["round=01"]
#                         )

#                 # save all patients scores in "inference_test_inter.json"
#                 Json.save(
#                     data=epoch_scores,
#                     path=os.path.join(epoch_dir, "inference_test_inter.json"),
#                 )
#                 continue  # next epoch dir

#     def remove_non_optimal_epochs(self, idl_gtvs_id: str, dataset: str = "valid"):
#         self._remove_non_optimal_epochs(train_id=idl_gtvs_id, dataset=dataset)
