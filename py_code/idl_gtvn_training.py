from custom import Global as g
import os
import torch
import math
import statistics
from tqdm import tqdm
from datetime import datetime
from baseline_training import BaselineTraining
from loss_func import UnifiedFocalLoss
from idl_gtvn_dataset import IDLGTVnDataSet
from torch.utils.data import DataLoader
from pathlib import Path
from custom import Json
from custom import List
from custom import Nii
from custom import Folder
from custom import GPU
from custom import Value
from custom import Dict
from custom import Explorer


class IDLGTVnTraining(BaselineTraining):
    def __load_hyper(
        self,
        hyper: Dict,
        fold: int,
        pred_main_folder: str,
        cnn_path: str = None,
        debug_mode: bool = False,
    ):
        # cross valid folds
        hyper["dataset.k.folds"] = g.DATASET_K_FOLDS

        # epochs
        if debug_mode:
            # at least 2 epochs to compare loss difference
            hyper["epochs"] = 2
        else:
            hyper["epochs"] = Value.limit_range(hyper["epochs"], (1, None))

        # record actual epochs because of early stop
        hyper["epochs.actual"] = 0

        # early stop, based on epoch
        hyper["early.stop.epochs"] = Value.limit_range(
            hyper["early.stop.epochs"], (1, hyper["epochs"])
        )

        # lr
        hyper["lr"] = Value.limit_range(hyper["lr"], (g.EPS, 1.0))

        # actual lr
        if GPU.used_count() > 1:
            hyper["lr.actual"] = hyper["lr"] * GPU.used_count()
        else:
            hyper["lr.actual"] = hyper["lr"]

        # min lr
        hyper["lr.min"] = Value.limit_range(hyper["lr.min"], (g.EPS, hyper["lr"]))

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        hyper["lr.decay.patience"] = Value.limit_range(
            hyper["lr.decay.patience"], (1, hyper["epochs"])
        )

        # number of best valid loss cnn retained
        hyper["keep.best.cnn.num"] = Value.limit_range(
            hyper["keep.best.cnn.num"], (1, hyper["epochs"])
        )

        # augment percent
        hyper["augment.pct"] = Value.limit_range(hyper["augment.pct"], (0.0, 1.0))

        # load shared hyper parameters
        super()._load_hyper(hyper=hyper, cnn_path=cnn_path)

        # redefine loss function
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
            training_type="idl_gtvn",
        ).to(g.DEVICE)

        # load patients
        (train_patients, valid_patients, test_patients,) = self._load_dataset(
            fold=fold,
            debug_mode=debug_mode,
        )

        # create datasets
        # run this after shared hyper loaded, because hyper["augment"] is needed
        augment = Dict()
        augment["methods"] = hyper["augment.methods"]
        augment["pct"] = hyper["augment.pct"]
        augment["min"] = hyper["augment.min"]
        augment["max"] = hyper["augment.max"]

        train_set = IDLGTVnDataSet(
            patient_list=train_patients,
            pred_main_folder=os.path.join(pred_main_folder, "train"),
            augment=augment,
        )
        valid_set = IDLGTVnDataSet(
            patient_list=valid_patients,
            pred_main_folder=os.path.join(pred_main_folder, "train"),
        )
        test_set = IDLGTVnDataSet(
            patient_list=test_patients,
            pred_main_folder=os.path.join(pred_main_folder, "test"),
        )

        # dataloader
        hyper["train.loader"] = DataLoader(
            dataset=train_set,
            batch_size=hyper["batch.size.actual"],
            shuffle=True,  # only shuffle train loader
            num_workers=g.NUM_WORKERS,
        )
        hyper["valid.loader"] = DataLoader(
            dataset=valid_set,
            batch_size=hyper["batch.size.actual"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )
        hyper["test.loader"] = DataLoader(
            dataset=test_set,
            batch_size=hyper["batch.size.actual"],
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )

    def __training(self, hyper: Dict, cur_fold_folder: str):
        best_loss_dict = Dict()
        loss_save_path = os.path.join(cur_fold_folder, "hyper", "loss.json")
        lr_save_path = os.path.join(cur_fold_folder, "hyper", "lr.json")
        patience = 0

        for cur_epoch in range(1, hyper["epochs"] + 1):
            print("")
            print("epoch: {}".format(cur_epoch))
            print("training:")
            hyper["cnn"].train()
            sum_loss = 0
            num_batches = 0
            for inputs, labels in tqdm(hyper["train.loader"]):
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = hyper["cnn"](inputs)
                loss = hyper["loss.func"](outputs, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                sum_loss += loss.item()
                num_batches += 1
            train_loss = sum_loss / num_batches

            # validation
            print("validation:")
            hyper["cnn"].eval()
            with torch.no_grad():
                sum_loss = 0
                num_batches = 0
                for inputs, labels in tqdm(hyper["valid.loader"]):
                    inputs = inputs.to(g.DEVICE)
                    labels = labels.to(g.DEVICE)
                    outputs = hyper["cnn"](inputs)
                    loss = hyper["loss.func"](outputs, labels)
                    sum_loss += loss.item()
                    num_batches += 1
            valid_loss = sum_loss / num_batches
            hyper["scheduler"].step(valid_loss)

            # current epoch finished
            hyper["epochs.actual"] = cur_epoch

            # save loss in json
            loss_dict = Json.load(loss_save_path)
            cur_epoch_loss = Dict()
            cur_epoch_loss["train"] = train_loss
            cur_epoch_loss["valid"] = valid_loss
            loss_dict["epoch={:03d}".format(hyper["epochs.actual"])] = cur_epoch_loss
            Json.save(loss_dict, loss_save_path)
            # draw loss figure
            self._plot_loss_fig(loss_save_path)

            # save lr in json
            lr_dict = Json.load(lr_save_path)
            for param_group in hyper["optim"].param_groups:
                cur_epoch_lr = param_group["lr"]
            lr_dict["epoch={:03d}".format(hyper["epochs.actual"])] = cur_epoch_lr
            Json.save(lr_dict, lr_save_path)
            # draw lr figure
            self._plot_lr_fig(lr_save_path)

            # save cnn
            if len(best_loss_dict) < hyper["keep.best.cnn.num"]:
                best_loss_dict[cur_epoch] = valid_loss
                cur_epoch_folder = os.path.join(
                    cur_fold_folder, "epoch={:03d}".format(cur_epoch)
                )
                Folder.create(cur_epoch_folder)
                cnn_save_path = os.path.join(
                    cur_epoch_folder,
                    "epoch={:03d}.pt".format(cur_epoch),
                )
                self._save_cnn(hyper, cnn_save_path)
            else:
                worst_epoch = best_loss_dict.key_with_max_value()
                worst_loss = best_loss_dict[worst_epoch]
                if valid_loss < worst_loss:
                    Folder.delete(
                        os.path.join(
                            cur_fold_folder, "epoch={:03d}".format(worst_epoch)
                        )
                    )
                    best_loss_dict.pop(worst_epoch)
                    best_loss_dict[cur_epoch] = valid_loss
                    cur_epoch_folder = os.path.join(
                        cur_fold_folder, "epoch={:03d}".format(cur_epoch)
                    )
                    Folder.create(cur_epoch_folder)
                    cnn_save_path = os.path.join(
                        cur_epoch_folder,
                        "epoch={:03d}.pt".format(cur_epoch),
                    )
                    self._save_cnn(hyper, cnn_save_path)
                    patience = 0
                else:
                    patience += 1
                    if patience >= hyper["early.stop.epochs"]:
                        break

    def new_training(
        self,
        baseline_id: str,
        baseline_fold: int = None,
        baseline_epoch: int = None,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        for cur_hyper in self._load_hyper_list_from_json(g.HYPER_JSON_PATH_IDL_GTVN):

            idl_gtvn_id = "idl_gtvn_" + self._init_train_id(
                train_remark=train_remark,
                debug_mode=debug_mode,
                hyper_json_path=g.HYPER_JSON_PATH_IDL_GTVN,
                hyper=cur_hyper,
            )
            print("")
            print(idl_gtvn_id)

            # find fold folder
            if baseline_fold is None or baseline_fold <= 0:
                key_word = "fold="
            else:
                key_word = "fold={:02d}".format(baseline_fold)
            baseline_fold_folder = Explorer.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_DIR, baseline_id),
                key_word=key_word,
                return_full_path=True,
            )[0]

            # find epoch folder
            if baseline_epoch is None or baseline_epoch <= 0:
                key_word = "epoch="
            else:
                key_word = "epoch={:03d}".format(baseline_epoch)
            baseline_epoch_folder = Explorer.get_sub_folders(
                baseline_fold_folder, key_word=key_word, return_full_path=True
            )[0]

            pred_main_folder = os.path.join(
                baseline_epoch_folder, "baseline", "patients"
            )

            idl_gtvn_folder = os.path.join(
                baseline_epoch_folder, "idl_gtvn", idl_gtvn_id
            )
            Folder.create(idl_gtvn_folder)

            for cur_fold in range(1, g.DATASET_K_FOLDS + 1):
                cur_fold_folder = os.path.join(
                    idl_gtvn_folder, "fold={:02d}".format(cur_fold)
                )
                Folder.create(cur_fold_folder)

                self.__load_hyper(
                    hyper=cur_hyper,
                    fold=cur_fold,
                    pred_main_folder=pred_main_folder,
                    cnn_path=None,
                    debug_mode=debug_mode,
                )
                if cur_fold == 1:
                    print("")
                    self._print_hyper(cur_hyper)

                print("")
                print("cross validation fold: {}".format(cur_fold))

                cur_hyper_folder = os.path.join(cur_fold_folder, "hyper")
                Folder.create(cur_hyper_folder)
                # save an empty loss.json
                Json.save(Dict(), os.path.join(cur_hyper_folder, "loss.json"))
                # save an empty lr.json
                Json.save(Dict(), os.path.join(cur_hyper_folder, "lr.json"))

                # save hyper before training
                hyper_save_path = os.path.join(cur_hyper_folder, "hyper.json")
                self._save_hyper(cur_hyper, hyper_save_path)

                # start training
                cur_hyper["time.spent"] = datetime.now()
                self.__training(cur_hyper, cur_fold_folder)
                cur_hyper["time.spent"] = datetime.now() - cur_hyper["time.spent"]
                cur_hyper["time.spent"] = str(cur_hyper["time.spent"]).split(".", 2)[0]

                # save hyper after training
                self._save_hyper(cur_hyper, hyper_save_path)

                # clear time spent before next training
                cur_hyper.pop("time.spent")

                # break if no cross validation
                if cur_hyper["dataset.cross.valid"] is False:
                    break
                # only train 2 folds in debug mode
                if debug_mode and cur_fold == 2:
                    break

            # inference
            for dataset in ["valid", "test"]:
                self.inference(
                    idl_gtvn_id=idl_gtvn_id,
                    dataset=dataset,
                    debug_mode=debug_mode,
                )

    def inference(
        self,
        idl_gtvn_id: str,
        dataset: str = "test",
        debug_mode: bool = False,
    ):
        print("")
        print("inference on {} set: {}".format(dataset, idl_gtvn_id))

        # find idl gtvt folder
        idl_gtvn_folder = self._find_result_folder(idl_gtvn_id)
        if idl_gtvn_folder is None:
            print("idl_gtvn_id not found")
            return

        if dataset != "valid":
            dataset = "test"

        if dataset == "valid":
            best_scores = Dict()

        baseline_epoch_folder = str(Path(idl_gtvn_folder).parent.parent)
        pred_main_folder = os.path.join(
            baseline_epoch_folder, "baseline", "patients", dataset
        )

        # loop through fold folders
        for cur_fold_folder in Explorer.get_sub_folders(
            idl_gtvn_folder, key_word="fold="
        ):
            cur_fold = int(cur_fold_folder[len("fold=") :])
            print("")
            print("current fold: ", cur_fold)
            cur_fold_folder = os.path.join(idl_gtvn_folder, cur_fold_folder)

            # loop through epoch folders
            for cur_epoch_folder in Explorer.get_sub_folders(
                cur_fold_folder, key_word="epoch="
            ):
                cur_epoch = int(cur_epoch_folder[len("epoch=") :])
                print("current epoch: ", cur_epoch)
                cur_epoch_folder = os.path.join(cur_fold_folder, cur_epoch_folder)

                # initialize median score
                cur_score = Dict()
                for metric in g.METRICS:
                    cur_score["median"][metric] = List()

                # load cnn
                cur_cnn_path = Explorer.get_sub_files(
                    cur_epoch_folder,
                    key_word=".pt",
                    return_full_path=True,
                )[0]
                cur_hyper = Dict()  # create an empty hyper dict to save cnn
                self._load_cnn(hyper=cur_hyper, cnn_path=cur_cnn_path)

                # load dataset
                _, valid_patients, test_patients = self._load_dataset(
                    fold=cur_fold, debug_mode=debug_mode
                )
                if dataset == "test":
                    patient_list = test_patients
                else:
                    patient_list = valid_patients

                # copy baseline score
                baseline_score = Json.load(
                    os.path.join(
                        baseline_epoch_folder,
                        "baseline",
                        "score_{}.json".format(dataset),
                    )
                )
                idl_gtvn_score_path = os.path.join(
                    cur_epoch_folder, "score_{}.json".format(dataset)
                )
                idl_gtvn_score = Dict()
                for cur_patient in patient_list:
                    for metric in g.METRICS:
                        idl_gtvn_score["patient={}".format(cur_patient)][metric][
                            "baseline"
                        ] = baseline_score["patient={}".format(cur_patient)]["gtvn"][
                            metric
                        ]
                    Json.save(idl_gtvn_score, idl_gtvn_score_path)

                for cur_patient in tqdm(patient_list):
                    # if on testset, create folder to save cur patient preds
                    if dataset == "test":
                        cur_patient_folder = os.path.join(
                            cur_epoch_folder,
                            "patients",
                            "patient={}".format(cur_patient),
                        )
                        Folder.create(cur_patient_folder)

                    # result structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    cur_patient_result = self._patient_inference(
                        patient=cur_patient,
                        hyper=cur_hyper,
                        inference_type="idl_gtvn",
                        pred_main_folder=pred_main_folder,
                    )

                    # save score of cur patient
                    for metric in g.METRICS:
                        # add cur patient result in avg_list
                        cur_score["median"][metric].append(
                            cur_patient_result["gtvn"][metric]
                        )
                        # record cur patient result if on test test
                        if dataset == "test":
                            cur_score["patient={}".format(cur_patient)][
                                metric
                            ] = cur_patient_result["gtvn"][metric]

                    # save pred of cur patient
                    if dataset == "test":
                        Nii.save(
                            img=cur_patient_result["gtvn"]["pred"],
                            path=os.path.join(
                                cur_patient_folder, "pred_{}.nii".format("gtvn")
                            ),
                            spacing=g.NII_SPACING,
                        )

                # calculate median score
                for metric in g.METRICS:
                    median = cur_score["median"][metric]
                    cur_score["median"][metric] = statistics.median(median)

                # save score (test set)
                if dataset == "test":
                    Json.save(data=cur_score, path=idl_gtvn_score_path)
                    continue

                # valid set, delete non-optimal folds and epochs
                if math.isnan(cur_score["median"]["msd"]) or math.isnan(
                    cur_score["median"]["hd95"]
                ):
                    # Folder.delete(cur_epoch_folder)
                    continue

                if best_scores == {}:
                    save_cur_score = True
                else:
                    save_cur_score = None
                    # loop through best_scores
                    for fold in best_scores.keys():
                        for epoch in best_scores[fold].keys():
                            # cur_score is worse than best_score, dont save
                            if (
                                cur_score["median"]["dsc"]
                                < best_scores[fold][epoch]["dsc"]
                                and cur_score["median"]["msd"]
                                > best_scores[fold][epoch]["msd"]
                                and cur_score["median"]["hd95"]
                                > best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = False
                                break
                            # cur_score is better than best_score, save
                            # (at least one of dsc/msd/hd95 is better)
                            if (
                                cur_score["median"]["dsc"]
                                > best_scores[fold][epoch]["dsc"]
                                or cur_score["median"]["msd"]
                                < best_scores[fold][epoch]["msd"]
                                or cur_score["median"]["hd95"]
                                < best_scores[fold][epoch]["hd95"]
                            ):
                                save_cur_score = True
                            # best_score is worse than cur score, delete best_score
                            if (
                                best_scores[fold][epoch]["dsc"]
                                < cur_score["median"]["dsc"]
                                and best_scores[fold][epoch]["msd"]
                                > cur_score["median"]["msd"]
                                and best_scores[fold][epoch]["hd95"]
                                > cur_score["median"]["hd95"]
                            ):
                                pass
                                # Folder.delete(
                                #     os.path.join(
                                #         idl_gtvn_folder,
                                #         "fold={:02d}".format(fold),
                                #         "epoch={:03d}".format(epoch),
                                #     )
                                # )
                        if save_cur_score is False:
                            break

                # save cur avg score
                if save_cur_score:
                    best_scores[cur_fold][cur_epoch] = cur_score["median"]
                    Json.save(data=cur_score, path=idl_gtvn_score_path)
                else:
                    pass
                    # Folder.delete(cur_epoch_folder)
