import os
import torch
import numpy as np
from training_baseline import TrainingBaseline
from loss_func_idl_gtvn import UnifiedFocalLossIDLGTVn
from dataset_idl_gtvn import DataSetIDLGTVn
from custom import Global as g
from custom import Folder
from custom import Explorer
from custom import ValueUtils
from custom import Dict
from custom import List
from custom import Json
from custom import Img
from custom import Nii
from pathlib import Path
from tqdm import tqdm


class TrainingIDLGTVn(TrainingBaseline):
    def _load_unique_hyper(self, hyper: Dict, debug_mode: bool):
        # load cnn before _load_common_hyper, optimizer needs cnn
        self._load_cnn(hyper=hyper, in_chan=5, out_chan=2)

        self._load_common_hyper(hyper=hyper, debug_mode=debug_mode)

        # loss function
        hyper["loss.func"] = UnifiedFocalLossIDLGTVn(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

        # load train/valid/test datasets
        for i in ["train", "valid", "test.inter"]:
            # only use data augmentation on training set
            if i == "train":
                augment = Dict()
                augment["methods"] = hyper["augment.methods"]
                augment["pct"] = hyper["augment.pct"]
                augment["min"] = hyper["augment.min"]
                augment["max"] = hyper["augment.max"]
            else:
                augment = None
            hyper["{}.set".format(i)] = DataSetIDLGTVn(
                patients=hyper["{}.patients".format(i)],
                baseline_id=hyper["baseline.id"],
                augment=augment,
                random_click=False,
            )

        # load dataloader
        self._load_data_loader(hyper)

    def new_training(
        self,
        baseline_id: str,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        for hyper in self._load_hyper_sets_from_json(g.HYPER_JSON_PATH_IDL_GTVN):

            idl_gtvn_id = "idl.gtvn_" + self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.HYPER_JSON_PATH_IDL_GTVN,
                hyper=hyper,
                debug_mode=debug_mode,
            )
            print("")
            print(idl_gtvn_id)

            # create train result dir
            idl_gtvn_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, idl_gtvn_id)
            Folder.create(idl_gtvn_dir)

            hyper["baseline.id"] = baseline_id

            self._training_traverse_folds(
                hyper=hyper, train_dir=idl_gtvn_dir, debug_mode=debug_mode
            )

            # inference
            self.inference(idl_gtvn_id=idl_gtvn_id, debug_mode=debug_mode)

            # after inference
            self.remove_non_optimal_epochs(idl_gtvn_id)

            # after non optimal epochs removed
            self.calculate_cross_valid_mean(
                idl_gtvn_id=idl_gtvn_id, debug_mode=debug_mode
            )

    def calculate_cross_valid_mean(self, idl_gtvn_id: str, debug_mode: bool = False):
        print("")
        print("calculate cross valid mean: {}".format(idl_gtvn_id))

        idl_gtvn_dir = self._find_train_dir(idl_gtvn_id)

        cross_valid_dir = os.path.join(idl_gtvn_dir, "cross_valid")
        Folder.create(cross_valid_dir, "patients")

        fold_dirs = Explorer.get_sub_folders(
            idl_gtvn_dir, key_word="fold=", full_path=True
        )

        # initialize scores dict
        scores = Dict()
        for stats in ["median", "avg"]:
            for metric in g.METRICS:
                scores[stats][metric] = List()

        # load patients
        inter_test_patients = self._load_patients(debug_mode=debug_mode)["test.inter"]

        for patient in tqdm(inter_test_patients):
            # initialize preds
            pred = None

            for fold_dir in fold_dirs:
                # find epoch dir
                epoch_dirs = Explorer.get_sub_folders(
                    fold_dir, key_word="epoch=", full_path=True
                )
                if len(epoch_dirs) > 1:
                    self.remove_non_optimal_epochs(idl_gtvn_id)
                    epoch_dir = Explorer.get_sub_folders(
                        fold_dir, key_word="epoch=", full_path=True
                    )[0]
                else:
                    epoch_dir = epoch_dirs[0]

                # load preds
                patient_dir = os.path.join(
                    epoch_dir, "patients", "patient={}".format(patient)
                )
                img = Nii.load(os.path.join(patient_dir, "gtvn_pred.nii"))
                if pred is None:
                    pred = img
                else:
                    pred += img

            pred /= len(fold_dirs)

            # save cross_valid preds
            patient_dir = os.path.join(
                cross_valid_dir, "patients", "patient={}".format(patient)
            )
            Folder.create(patient_dir)
            Nii.save(img=pred, save_path=os.path.join(patient_dir, "gtvn_pred.nii"))

            # load labels and calculate metrics (on internal test set only)
            if patient in inter_test_patients:
                label = Nii.load(
                    os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVn.nii".format(patient)),
                    binary=True,
                )
                for metric in g.METRICS:
                    score = self._metrics[metric](pred, label)
                    scores["patient={}".format(patient)][metric] = score

                    for stats in ["median", "avg"]:
                        scores[stats][metric].append(score)

        # calculate avg and median score
        for metric in g.METRICS:
            scores["median"][metric] = ValueUtils.median(scores["median"][metric])
            scores["avg"][metric] = ValueUtils.avg(scores["avg"][metric])
        # save cross validscores
        Json.save(
            data=scores,
            path=os.path.join(cross_valid_dir, "inference_test_inter.json"),
        )

    def remove_non_optimal_epochs(self, idl_gtvn_id: str):
        idl_gtvn_dir = self._find_train_dir(idl_gtvn_id)
        print("")
        print("remove non optimal epochs:{}".format(idl_gtvn_dir))

        for fold_dir in Explorer.get_sub_folders(
            idl_gtvn_dir, key_word="fold=", full_path=True
        ):
            fold_scores = Dict()

            for epoch_dir in Explorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name

                # load scores of current epoch
                # if baseline
                epoch_scores = Json.load(
                    os.path.join(epoch_dir, "inference_test_inter.json")
                )
                for stats in ["median", "avg"]:
                    for metric in g.METRICS:
                        fold_scores[epoch][stats][metric] = epoch_scores[stats][metric][
                            "round=01"
                        ]

            best_epoch = self.__find_best_result(fold_scores)

            # delete non-optimal epochs
            for epoch_dir in Explorer.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                if epoch != best_epoch:
                    Folder.delete(epoch_dir)
                    print("delete:", epoch_dir)

    # a sub function of _remove_non_optimal_epochs()
    def __find_best_result(self, scores: Dict):
        for stats in ["median", "avg"]:
            for metric in g.METRICS:
                # create a tmp list to sort
                list_to_sort = List()
                # add elements into the list
                for epoch in scores.keys():
                    list_to_sort.append(scores[epoch][stats][metric])
                # sort the list
                if metric == "dsc":
                    list_to_sort.sort(reverse=False)
                else:
                    list_to_sort.sort(reverse=True)
                # update value based on the idx in the list
                for epoch in scores.keys():
                    new_value = list_to_sort.index(scores[epoch][stats][metric])
                    # if metric == "dsc":
                    #     new_value *= 2
                    scores[epoch][stats][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stats in ["avg", "median"]:
                for metric in g.METRICS:
                    evaluation[epoch] += scores[epoch][stats][metric]

        return evaluation.key_with_max_value()

    def __single_patient_inference(
        self, patient: str, hyper: Dict, baseline_id: str
    ) -> Dict:

        result = Dict()  # ["gtvn"]["label/pred"]

        dataset = DataSetIDLGTVn(
            patients=[patient],
            baseline_id=baseline_id,
            augment=None,
            random_click=False,
        )

        # load label
        result["gtvn"]["label"] = Nii.load(
            os.path.join(g.DATASET_DIR, "HNCDL_{}_GTVn.nii".format(patient)),
            binary=True,
        )

        # get pred
        input_imgs, labels, gtvn_clicks = dataset.get_item(patient)

        # add "batch" (c/d/h/w -> b/c/d/h/w)
        input_imgs = torch.unsqueeze(input_imgs.to(g.DEVICE), dim=0)
        labels = torch.unsqueeze(labels.to(g.DEVICE), dim=0)

        hyper["cnn"].eval()  # disable dropout / batch nomalize
        with torch.no_grad():
            preds = hyper["cnn"].forward(input_imgs)
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        preds = torch.squeeze(preds, dim=0).cpu().numpy()

        result["gtvn"]["pred"] = preds[1]
        # squeeze "batch" (b/c/d/h/w -> c/d/h/w)
        input_imgs = torch.squeeze(input_imgs, dim=0).cpu().numpy()
        result["gtvn"]["distance.map"] = input_imgs[0]
        result["gtvn"]["clicks"] = torch.squeeze(gtvn_clicks, dim=0).cpu().numpy()

        # pad and crop to original size
        for i in ["pred", "distance.map", "clicks"]:
            result["gtvn"][i] = Img.central_pad_and_crop(
                result["gtvn"][i], result["gtvn"]["label"].shape
            )

        # idl_gtvn post processing
        if 0:
            cc_list = Img.connected_components(result["gtvn"]["pred"])
            result["gtvn"]["pred"] = np.zeros_like(result["gtvn"]["pred"])
            for cur_cc in cc_list:
                if (cur_cc * result["gtvn"]["clicks"]).sum() > 0:
                    result["gtvn"]["pred"] = np.maximum(result["gtvn"]["pred"], cur_cc)

        return result

    def inference(self, idl_gtvn_id: str, debug_mode: bool = False):
        print("")
        print("inference: {}".format(idl_gtvn_id))

        idl_gtvn_dir = self._find_train_dir(idl_gtvn_id)
        if idl_gtvn_dir is None:
            print("idl_gtvn_id not found")
            return

        # loop through fold dirs
        for fold_dir in Explorer.get_sub_folders(
            idl_gtvn_dir, key_word="fold=", full_path=True
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
                cnn_path = os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch))
                hyper = Dict()  # create an empty hyper dict to save cnn
                self._load_cnn(hyper=hyper, cnn_path=cnn_path)

                # load patients
                inter_test_patients = self._load_patients(debug_mode=debug_mode)[
                    "test.inter"
                ]

                # initialize scores dict (only on test and valid set)
                epoch_scores = Dict()

                # initialize ["round=01"] as a list
                for stats in ["median", "avg"]:
                    for metric in g.METRICS:
                        epoch_scores[stats][metric]["round=01"] = List()

                # load baseline scores
                baseline_scores = Json.load(
                    os.path.join(
                        Path(idl_gtvn_dir).parent,
                        "baseline",
                        "cross_valid",
                        "inference_test_inter.json",
                    )
                )
                # copy baseline scores of each patient
                for patient in inter_test_patients:
                    for metric in g.METRICS:
                        epoch_scores["patient={}".format(patient)][metric][
                            "round=00"
                        ] = baseline_scores["patient={}".format(patient)]["gtvn"][
                            metric
                        ]
                # also copy median score of each patient
                for stats in ["median", "avg"]:
                    for metric in g.METRICS:
                        epoch_scores[stats][metric]["round=00"] = baseline_scores[
                            stats
                        ]["gtvn"][metric]

                # loop through each patient
                for patient in tqdm(inter_test_patients):
                    # create folder to save cur patient preds and scores
                    patient_dir = os.path.join(
                        epoch_dir,
                        "patients",
                        "patient={}".format(patient),
                    )
                    Folder.create(patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self.__single_patient_inference(
                        patient=patient,
                        hyper=hyper,
                        baseline_id=Path(idl_gtvn_dir).parent.name,
                    )

                    # save preds of current patient
                    for i in ["pred", "distance.map", "clicks"]:
                        Nii.save(
                            img=patient_results["gtvn"]["distance.map"],
                            save_path=os.path.join(
                                patient_dir, "gtvn_{}.nii".format(i.replace(".", "_"))
                            ),
                            spacing=g.NII_SPACING,
                        )

                    # record score of current patient
                    for metric in g.METRICS:
                        score = self._metrics[metric](
                            patient_results["gtvn"]["pred"],
                            patient_results["gtvn"]["label"],
                        )
                        # save cur patient score
                        epoch_scores["patient={}".format(patient)][metric][
                            "round=01"
                        ] = score
                        # add scores of current patient into median(list)
                        # record in ["round=01"] for test set
                        for stats in ["median", "avg"]:
                            epoch_scores[stats][metric]["round=01"].append(score)

                # all patients under current epoch have been traversed
                # calculate median and avg score of current epoch
                for metric in g.METRICS:
                    epoch_scores["median"][metric]["round=01"] = ValueUtils.median(
                        epoch_scores["median"][metric]["round=01"]
                    )
                    epoch_scores["avg"][metric]["round=01"] = ValueUtils.avg(
                        epoch_scores["avg"][metric]["round=01"]
                    )
                # save all patients scores in "inference_test_inter.json"
                Json.save(
                    data=epoch_scores,
                    path=os.path.join(epoch_dir, "inference_test_inter.json"),
                )
                continue  # next epoch
