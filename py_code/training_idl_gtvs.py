import os
import torch
import numpy as np
from training_baseline import TrainingBaseline
from dataset_idl_gtvs import DataSetIDLGTVs
from custom import Folder
from custom import Json
from custom import Explorer
from custom import Nii
from custom import Dict
from custom import List
from custom import Img
from custom import GPU
from custom import ValueUtils
from pathlib import Path
from tqdm import tqdm
from custom import Global as g
from torch.nn import DataParallel
from unet_pp_slim import UNetPPSlim


class TrainingIDLGTVs(TrainingBaseline):
    # if float64 needed, use: "cnn.to(torch.double)"
    def __load_cnn(self, hyper: Dict, cnn_path: str = None):
        # new model
        if cnn_path == "" or cnn_path is None:
            hyper["cnn"] = UNetPPSlim(
                in_chan=6, out_chan=3, dropout=hyper["dropout"]
            ).to(g.DEVICE)

        # existing model
        else:
            hyper["cnn"] = torch.load(cnn_path).to(g.DEVICE)

        # set multi-GPU
        if GPU.used_count() > 1:
            hyper["cnn"] = DataParallel(hyper["cnn"]).to(g.DEVICE)

    def new_training(
        self,
        baseline_id: str = None,
        baseline_fold: int = None,
        baseline_epoch: int = None,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        self._new_training(
            train_type="idl",
            baseline_id=baseline_id,
            baseline_fold=baseline_fold,
            baseline_epoch=baseline_epoch,
            train_remark=train_remark,
            debug_mode=debug_mode,
        )

    def __patient_inference(
        self,
        patient: str,
        hyper: Dict,
        baseline_epoch_dir: str = None,  # dataset needs this
    ) -> Dict:
        result = Dict()  # gtv->metric

        origin = Dict()  # original labels

        weight = Dict()
        weight["background"] = 0.2
        weight["distance.step"] = 2
        weight["fp.fn"] = 1
        weight["prev.round.decay"] = 0.5
        weight["slice"] = 1
        dataset = DataSetIDLGTVs(
            patients=[patient],
            baseline_epoch_dir=baseline_epoch_dir,
            weight=weight,
            random_click=False,
        )

        # load labels
        for gtv in ["s", "t", "n"]:
            origin["gtv{}".format(gtv)] = Nii.load(
                os.path.join(g.DATASET_DIR, "HNCDL_{}_GTV{}.nii".format(patient, gtv)),
                binary=True,
            )

        # get pred
        input_imgs, labels, gtvt_weight_map, gtvn_clicks = dataset.get_item(
            patient=patient
        )
        # add "batch" (c/d/h/w -> b/c/d/h/w)
        input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)
        labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)
        hyper["cnn"].eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            preds = hyper["cnn"].forward(input_imgs)
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        preds = torch.squeeze(preds, dim=0).cpu().numpy()

        result["gtvt"]["pred"] = preds[1]
        result["gtvn"]["pred"] = preds[2]
        result["gtvs"]["pred"] = np.maximum(preds[1], preds[2])

        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        input_imgs = torch.squeeze(input_imgs, dim=0).cpu().numpy()
        result["gtvt"]["annotation"] = input_imgs[0]
        result["gtvn"]["distance.map"] = input_imgs[1]
        # squeeze "channel" (c/d/h/w -> d/h/w)
        result["gtvt"]["weight.map"] = (
            torch.squeeze(gtvt_weight_map, dim=0).cpu().numpy()
        )
        result["gtvn"]["clicks"] = torch.squeeze(gtvn_clicks, dim=0).cpu().numpy()

        # pad and crop to original size
        # 1.preds
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            result[gtv]["pred"] = Img.central_pad_and_crop(
                result[gtv]["pred"], origin["gtvs"].shape
            )
        # 2.annotation and weight_map
        for i in ["annotation", "weight.map"]:
            result["gtvt"][i] = Img.central_pad_and_crop(
                result["gtvt"][i], origin["gtvs"].shape
            )
        # 3.distance_map and clicks
        for i in ["distance.map", "clicks"]:
            result["gtvn"][i] = Img.central_pad_and_crop(
                result["gtvn"][i], origin["gtvs"].shape
            )

        # gtvt post processing
        if 1:
            cc_list = Img.connected_components(result["gtvt"]["pred"])
            result["gtvt"]["pred"] = np.zeros_like(result["gtvt"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * result["gtvt"]["annotation"]).sum() > 0:
                    result["gtvt"]["pred"] = np.maximum(result["gtvt"]["pred"], cur_cc)

        # idl_gtvn post processing
        if 0:
            cc_list = Img.connected_components(result["gtvn"]["pred"])
            result["gtvn"]["pred"] = np.zeros_like(result["gtvn"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * result["gtvn"]["clicks"]).sum() > 0:
                    result["gtvn"]["pred"] = np.maximum(result["gtvn"]["pred"], cur_cc)

        # calculate inference scores
        for gtv in ["gtvs", "gtvt", "gtvn"]:
            for metric in g.METRICS:
                result[gtv][metric] = self._metrics[metric](
                    result[gtv]["pred"], origin[gtv]
                )
        return result

    def inference(
        self,
        idl_id: str,
        debug_mode: bool = False,
    ):
        print("")
        print("inference: {}".format(idl_id))

        # find idl folder
        idl_result_dir = self._find_result_dir(idl_id)
        if idl_result_dir is None:
            print("idl_id not found")
            return

        baseline_epoch_dir = str(Path(idl_result_dir).parent.parent)

        # loop through fold dirs
        for fold_dir in Explorer.get_sub_folders(
            idl_result_dir, key_word="fold=", full_path=True
        ):
            fold = int(Path(fold_dir).name[len("fold=") :])
            print("")
            print("fold: ", fold)

            # loop through epoch dirs
            for epoch_dir in Explorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("epoch: ", epoch)

                # load cnn
                cnn_path = Explorer.get_sub_files(
                    epoch_dir, key_word=".pt", full_path=True
                )[0]
                hyper = Dict()  # create an empty dict to save cnn
                hyper["dropout"] = 0
                self.__load_cnn(hyper=hyper, cnn_path=cnn_path)

                # load test patients
                patients = self._load_dataset(fold=fold, debug_mode=debug_mode)[2]

                # initialize scores dict
                epoch_scores = Dict()
                gtv_list = ["gtvs", "gtvt", "gtvn"]

                # copy baseline scores
                baseline_scores = Json.load(
                    os.path.join(baseline_epoch_dir, "baseline", "inference.json")
                )
                key_list = ["median", "avg"]
                for patient in patients:
                    key_list.append("patient={}".format(patient))
                for i in key_list:
                    for gtv in ["gtvs", "gtvt", "gtvn"]:
                        for metric in g.METRICS:
                            epoch_scores[i][gtv][metric]["round=00"] = baseline_scores[
                                i
                            ][gtv][metric]

                # initialize median and avg score of round=01
                for i in ["median", "avg"]:
                    for gtv in gtv_list:
                        for metric in g.METRICS:
                            epoch_scores[i][gtv][metric]["round=01"] = List()

                # loop through each patient
                for patient in tqdm(patients):
                    # create folder to save cur patient preds and scores
                    patient_dir = os.path.join(
                        epoch_dir,
                        "patients",
                        "patient={}".format(patient),
                    )
                    Folder.create(patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self.__patient_inference(
                        patient=patient,
                        hyper=hyper,
                        baseline_epoch_dir=baseline_epoch_dir,
                    )

                    # save Niis of current patient
                    Nii.save(
                        img=patient_results["gtvt"]["annotation"],
                        path=os.path.join(patient_dir, "gtvt_annotation.nii"),
                        spacing=g.NII_SPACING,
                    )
                    Nii.save(
                        img=patient_results["gtvt"]["weight.map"],
                        path=os.path.join(patient_dir, "gtvt_weight_map.nii"),
                        spacing=g.NII_SPACING,
                    )
                    Nii.save(
                        img=patient_results["gtvn"]["distance.map"],
                        path=os.path.join(patient_dir, "gtvn_distance_map.nii"),
                        spacing=g.NII_SPACING,
                    )
                    Nii.save(
                        img=patient_results["gtvn"]["clicks"],
                        path=os.path.join(patient_dir, "gtvn_clicks.nii"),
                        spacing=g.NII_SPACING,
                    )
                    for gtv in gtv_list:
                        Nii.save(
                            img=patient_results[gtv]["pred"],
                            path=os.path.join(patient_dir, "{}_pred.nii".format(gtv)),
                            spacing=g.NII_SPACING,
                        )

                    # record score of current patient
                    for gtv in gtv_list:
                        for metric in g.METRICS:
                            # save cur patient score into inference_test.json (test set only)
                            epoch_scores["patient={}".format(patient)][gtv][metric][
                                "round=01"
                            ] = patient_results[gtv][metric]
                            # add scores of current patient into median(list)
                            for i in ["median", "avg"]:
                                epoch_scores[i][gtv][metric]["round=01"].append(
                                    patient_results[gtv][metric]
                                )

                # all patients under current epoch have been traversed
                # calculate avg and median score
                for gtv in gtv_list:
                    for metric in g.METRICS:
                        median = ValueUtils.median(
                            epoch_scores["median"][gtv][metric]["round=01"]
                        )
                        epoch_scores["median"][gtv][metric]["round=01"] = median

                        avg = ValueUtils.avg(
                            epoch_scores["avg"][gtv][metric]["round=01"]
                        )
                        epoch_scores["avg"][gtv][metric]["round=01"] = avg

                # save all patients scores in "inference.json"
                Json.save(
                    data=epoch_scores,
                    path=os.path.join(epoch_dir, "inference.json"),
                )
                continue  # next epoch dir

    def remove_non_optimal_epochs(self, idl_id: str, dataset: str = "valid"):
        self._remove_non_optimal_epochs(train_id=idl_id, dataset=dataset)
