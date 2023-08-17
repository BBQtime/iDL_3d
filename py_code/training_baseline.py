import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from custom import GPU, Debug, Dict, Directory, Folder
from custom import Global as g
from custom import Img, Json, List, Nii, Value
from dataset_baseline import DataSetBaseline
from loss_func import UnifiedFocalLoss
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_core import TrainingCore


class TrainingBaseline(TrainingCore):
    def _load_hyper(
        self,
        hyper: Dict,
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # shared by baseline/idl.gtvn/idl.gtvt
        super()._load_hyper(hyper)

        # epochs
        if debug_mode:
            # only train 2 epoch in debug mode
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

        self._load_hyper_dataset_version(hyper)

        # load patients after dataset version is selected
        patients = self._load_patients(
            dataset_ver=hyper["dataset.ver"], fold=hyper["fold"], debug_mode=debug_mode
        )
        for key_name in ["train", "valid"]:
            hyper["{}.patients".format(key_name)] = patients[key_name]

        self._load_hyper_loss_func(hyper)

        # load datasets before load dataloaders
        self._load_hyper_data_sets(hyper)

        self._load_hyper_data_loaders(hyper)

        # load cnn before optimizer
        self._load_hyper_new_cnn(hyper=hyper)

        self._load_hyper_optim_and_scheduler(hyper=hyper)

    def _load_hyper_new_cnn(self, hyper: Dict, in_chan: int = 4, out_chan: int = 3):
        super()._load_hyper_new_cnn(hyper=hyper, in_chan=in_chan, out_chan=out_chan)

    def _load_hyper_loss_func(self, hyper: Dict):
        hyper["loss.func"] = UnifiedFocalLoss(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

    def _load_hyper_data_sets(self, hyper: Dict):
        # load train/valid/test datasets
        for i in ["train", "valid"]:
            # only use data augmentation on training set
            if i == "train":
                augment = Dict()
                augment["methods"] = hyper["augment.methods"]
                augment["pct"] = hyper["augment.pct"]
                augment["min"] = hyper["augment.min"]
                augment["max"] = hyper["augment.max"]
            else:
                augment = None
            hyper["{}.set".format(i)] = DataSetBaseline(
                patients=hyper["{}.patients".format(i)],
                dataset_ver=hyper["dataset.ver"],
                augment=augment,
            )

    # baseline/idl.gtvn/idl.gtvs share this function
    def _load_hyper_data_loaders(self, hyper: Dict):
        for i in ["train", "valid"]:
            # only shuffle train loader
            if i == "train":
                shuffle = True
            else:
                shuffle = False
            hyper["{}.loader".format(i)] = DataLoader(
                dataset=hyper["{}.set".format(i)],
                batch_size=hyper["batch.size.actual"],
                shuffle=shuffle,
                num_workers=g.NUM_WORKERS,
            )

    def _simplify_hyper(self, hyper: Dict) -> Dict:
        # use simiple_hyper here, dont change origin hyper dict
        simple_hyper = super()._simplify_hyper(hyper)

        ignore_list = []
        for i in ["train", "valid", "test.inter", "test.exter", "test"]:
            ignore_list.append("{}.patients".format(i))
            ignore_list.append("{}.set".format(i))
            ignore_list.append("{}.loader".format(i))

        # here in this for loop, use "hyper" instead of "simple_hyper"
        # otherwise will cause error: dictionary changed size during iteration
        for key_name in hyper:
            if key_name in ignore_list:
                simple_hyper.pop(key_name)
            else:
                pass
        return simple_hyper

    def _print_hyper(self, hyper: Dict):
        simple_hyper = self._simplify_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def _save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self._simplify_hyper(hyper)
        Json.save(data=simple_hyper, path=json_path)

    # protected function, TrainingIDLGTVn will inherit it
    def _plot_lr_fig(self, lr_json_path: str):
        plt.figure().clear()

        lr_dict = Json.load(lr_json_path)
        lr_list = List()
        for i in lr_dict:
            lr_list.append(lr_dict[i])

        plt.plot(range(1, len(lr_list) + 1), lr_list, label="lr")
        plt.legend()
        plt.savefig(lr_json_path[:-4] + "png")

    # protected function, TrainingIDLGTVn will inherit it
    def _plot_loss_fig(self, loss_json_path: str):
        loss_dict = Json.load(loss_json_path)
        train_loss = List()
        valid_loss = List()

        for i in loss_dict:
            train_loss.append(loss_dict[i]["train"])
            valid_loss.append(loss_dict[i]["valid"])

        # draw figure
        plt.figure().clear()
        plt.ylim(min(train_loss) - 0.05, max(train_loss) + 0.05)
        plt.plot(range(1, len(loss_dict) + 1), train_loss, label="train")
        plt.plot(range(1, len(loss_dict) + 1), valid_loss, label="valid")
        plt.legend()
        plt.savefig(loss_json_path[:-4] + "png")

    def _calculate_loss(self, item: tuple, hyper: Dict):
        input_imgs = item[0].to(g.DEVICE)
        labels = item[1].to(g.DEVICE)
        preds = hyper["cnn"](input_imgs)
        loss = hyper["loss.func"](preds, labels)
        return loss

    def _training_all_epochs(self, hyper: Dict, fold_dir: str):
        best_loss_dict = Dict()
        loss_json_path = os.path.join(fold_dir, "loss.json")
        lr_json_path = os.path.join(fold_dir, "lr.json")
        patience = 0

        for epoch in range(1, hyper["epochs"] + 1):
            print("")
            print("epoch: {}".format(epoch))
            print("training:")
            hyper["cnn"].train()
            train_loss = 0
            batch_count = 0

            # training
            for item in tqdm(hyper["train.loader"]):
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                loss = self._calculate_loss(item=item, hyper=hyper)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                train_loss += loss.item()
                batch_count += 1
            train_loss /= batch_count

            # validation
            print("validation:")
            valid_loss = 0
            batch_count = 0
            hyper["cnn"].eval()
            with torch.no_grad():
                for item in tqdm(hyper["valid.loader"]):
                    loss = self._calculate_loss(item=item, hyper=hyper)
                    valid_loss += loss.item()
                    batch_count += 1
            valid_loss /= batch_count
            hyper["scheduler"].step(valid_loss)

            # current epoch finished
            hyper["epochs.actual"] = epoch

            # save loss in json
            loss_dict = Json.load(loss_json_path)
            epoch_loss = Dict()
            epoch_loss["train"] = train_loss
            epoch_loss["valid"] = valid_loss
            loss_dict["epoch={:03d}".format(hyper["epochs.actual"])] = epoch_loss
            Json.save(loss_dict, loss_json_path)
            # draw loss figure
            self._plot_loss_fig(loss_json_path)

            # save lr in json
            lr_dict = Json.load(lr_json_path)
            for param_group in hyper["optim"].param_groups:
                epoch_lr = param_group["lr"]
            lr_dict["epoch={:03d}".format(hyper["epochs.actual"])] = epoch_lr
            Json.save(lr_dict, lr_json_path)
            # draw lr figure
            self._plot_lr_fig(lr_json_path)

            # save cnn
            if len(best_loss_dict) < hyper["keep.best.cnn.num"]:
                best_loss_dict[epoch] = valid_loss
                epoch_dir = os.path.join(fold_dir, "epoch={:03d}".format(epoch))
                Folder.create(epoch_dir)
                self._save_cnn(
                    hyper,
                    os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch)),
                )
            else:
                worst_epoch = best_loss_dict.key_with_max_value()
                worst_loss = best_loss_dict[worst_epoch]
                if valid_loss < worst_loss:
                    Folder.delete(
                        os.path.join(fold_dir, "epoch={:03d}".format(worst_epoch))
                    )
                    best_loss_dict.pop(worst_epoch)
                    best_loss_dict[epoch] = valid_loss
                    epoch_dir = os.path.join(fold_dir, "epoch={:03d}".format(epoch))
                    Folder.create(epoch_dir)
                    self._save_cnn(
                        hyper,
                        os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch)),
                    )
                    patience = 0
                else:
                    patience += 1
                    if patience >= hyper["early.stop.epochs"]:
                        break

    def _training_all_folds(
        self,
        hyper: Dict,
        train_dir: str,
        debug_mode: bool = False,
    ):
        Folder.create(train_dir)

        # cross validation
        hyper["fold"] = int(hyper["fold"])
        hyper["fold"] = Value.limit_range(hyper["fold"], (0, g.DATASET_FOLDS))
        # fold=0 will activate cross validation
        if hyper["fold"] == 0:
            fold_list = List(range(1, g.DATASET_FOLDS + 1))
        else:
            fold_list = [hyper["fold"]]

        # backup origin hyper for resetting hyper on next fold
        # (after "fold" removed from hyper Dict)
        hyper.pop("fold")
        origin_hyper = hyper.copy()

        # loop through each fold
        for hyper["fold"] in fold_list:
            fold_dir = os.path.join(train_dir, "fold={}".format(hyper["fold"]))
            Folder.create(fold_dir)

            # load and print hyperparams
            self._load_hyper(hyper=hyper, debug_mode=debug_mode)
            print("")
            self._print_hyper(hyper)

            print("")
            print("fold: {}".format(hyper["fold"]))

            # save an empty loss.json
            Json.save(Dict(), os.path.join(fold_dir, "loss.json"))
            # save an empty lr.json
            Json.save(Dict(), os.path.join(fold_dir, "lr.json"))

            # save hyper before training
            hyper_save_path = os.path.join(fold_dir, "hyper.json")
            self._save_hyper(hyper, hyper_save_path)

            # start training
            hyper["time.spent"] = datetime.now()
            self._training_all_epochs(hyper, fold_dir)
            hyper["time.spent"] = datetime.now() - hyper["time.spent"]
            hyper["time.spent"] = str(hyper["time.spent"]).split(".", 2)[0]

            # save hyper after training
            self._save_hyper(hyper, hyper_save_path)

            # only train 2 folds in debug mode
            if (
                debug_mode
                and len(fold_list) > 1
                and fold_list.index(hyper["fold"]) == 1
            ):
                break

            # reset hyper before next fold
            hyper = origin_hyper.copy()

    def new_training(
        self,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        self._new_training(
            train_type="baseline", train_remark=train_remark, debug_mode=debug_mode
        )

    def _new_training(
        self,
        train_type: str,  # baseline/idl.gtvn
        baseline_id: str = None,  # this is only for idl.gtvn
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        if train_type != "baseline" and train_type != "idl.gtvn":
            Debug.error_exit("training type error")

        for hyper in self._load_hyper_sets_from_json(g.HYPER_JSON_PATH[train_type]):
            train_id = train_type + "_"
            train_id += self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.HYPER_JSON_PATH[train_type],
                hyper=hyper,
                debug_mode=debug_mode,
            )

            print("")
            print(train_id)

            if train_type == "baseline":
                train_dir = os.path.join(g.TRAIN_RESULTS_DIR, train_id, "baseline")
            elif train_type == "idl.gtvn":
                train_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id, train_id)

            # add baseline_id to hyper Dict, don't need extra param
            hyper["baseline.id"] = baseline_id

            self._training_all_folds(
                hyper=hyper, train_dir=train_dir, debug_mode=debug_mode
            )

            # inference
            Value.is_valid_dataset_version(hyper["dataset.ver"])
            if hyper["dataset.ver"] == "mda":
                dataset_section_list = ["valid", "test"]
            else:
                dataset_section_list = ["valid", "test.inter"]

            if train_type == "baseline":
                dataset_section_list.append("train")

            for dataset_section in dataset_section_list:
                self._inference(
                    inference_type=train_type,
                    train_id=train_id,
                    dataset_ver=hyper["dataset.ver"],
                    dataset_section=dataset_section,
                    debug_mode=debug_mode,
                )

            # remove non-optimal epochs after inference
            self._remove_non_optimal_epochs(
                inference_type=train_type, train_id=train_id
            )

            # cross validation evaluation after non optimal epochs removed
            dataset_section_list.remove("valid")
            for dataset_section in dataset_section_list:
                self._cross_valid_evaluation(
                    inference_type=train_type,
                    train_id=train_id,
                    dataset_ver=hyper["dataset.ver"],
                    dataset_section=dataset_section,
                    debug_mode=debug_mode,
                )

    def inference(
        self,
        baseline_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self._inference(
            inference_type="baseline",
            train_id=baseline_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def __inference_prepare(
        self,
        inference_type: str,  # baseline/idl.gtvn
        train_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
    ):
        if inference_type != "baseline" and inference_type != "idl.gtvn":
            Debug.error_exit("inference type error")

        train_dir = self._find_train_dir(train_id)
        if train_dir is None:
            Debug.error_exit("training id not found")

        fold_dirs = Directory.get_sub_folders(
            train_dir, key_word="fold=", full_path=True
        )

        # load dataset version
        dataset_ver_training = Json.load(os.path.join(fold_dirs[0], "hyper.json"))[
            "dataset.ver"
        ]
        if dataset_ver is None:
            dataset_ver = dataset_ver_training
        Value.is_valid_dataset_version(
            dataset_ver=dataset_ver,
            dataset_ver_baseline_or_training=dataset_ver_training,
        )
        Value.is_valid_dataset_section(
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            inference_type=inference_type,
        )
        print("dataset version: {}".format(dataset_ver))
        print("dataset section: {}".format(dataset_section))

        # load segmentation metrics
        segment_metrics = self._load_segment_metrics(dataset_ver)

        return dataset_ver, fold_dirs, segment_metrics

    def __inference_init_scores(
        self,
        inference_type: str,
        baseline_id: str,
        dataset_ver: str,
        dataset_section: str,
        patients: Dict,
    ) -> Dict:
        scores = Dict()
        # baseline
        if inference_type == "baseline":
            for stats in ["median", "avg"]:
                for gtv in ["gtvs", "gtvt", "gtvn"]:
                    for metric in g.METRICS:
                        scores[stats][gtv][metric] = List()
        # idl.gtvn
        else:
            for stats in ["median", "avg"]:
                for metric in g.METRICS:
                    scores[stats][metric]["round=01"] = List()

            # only load baseline scores of test set, because there is no validation scores
            if "test" in dataset_section:
                # load baseline scores
                baseline_scores = Json.load(
                    os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        baseline_id,
                        "baseline",
                        "inference_{}_{}.json".format(dataset_ver, dataset_section),
                    )
                )

                # copy baseline gtvn scores of each patient
                for patient in patients[dataset_section]:
                    for metric in g.METRICS:
                        scores["patient={}".format(patient)][metric][
                            "round=00"
                        ] = baseline_scores["patient={}".format(patient)]["gtvn"][
                            metric
                        ]

                # also copy baseline median and avg gtvn scores
                for stats in ["median", "avg"]:
                    for metric in g.METRICS:
                        scores[stats][metric]["round=00"] = baseline_scores[stats][
                            "gtvn"
                        ][metric]

        return scores

    def _inference(
        self,
        inference_type: str,  # baseline/idl.gtvn
        train_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        if not train_id.startswith(inference_type):
            Debug.error_exit("train id error")
        print("")
        print("inference: {}".format(train_id))

        dataset_ver, fold_dirs, segment_metrics = self.__inference_prepare(
            inference_type=inference_type,
            train_id=train_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
        )

        # loop through fold dirs
        for fold_dir in fold_dirs:
            fold = int(Path(fold_dir).name[len("fold=") :])
            print("")
            print("fold: ", fold)

            # load patients
            patients = self._load_patients(
                dataset_ver=dataset_ver, fold=fold, debug_mode=debug_mode
            )

            # loop through epoch dirs
            for epoch_dir in Directory.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = int(Path(epoch_dir).name[len("epoch=") :])
                print("epoch: ", epoch)

                # load cnn
                cnn_path = os.path.join(epoch_dir, "epoch={:03d}.pt".format(epoch))
                cnn = self._load_exist_cnn(cnn_path)

                # initialize scores dict (only for test sets)
                if "test" in dataset_section or "valid" in dataset_section:
                    epoch_scores = self.__inference_init_scores(
                        inference_type=inference_type,
                        baseline_id=Path(fold_dir).parent.parent.name,
                        dataset_ver=dataset_ver,
                        dataset_section=dataset_section,
                        patients=patients,
                    )

                # loop through each patient
                for patient in tqdm(patients[dataset_section]):
                    # create folder to save cur patient preds and scores
                    epoch_patient_dir = os.path.join(
                        epoch_dir,
                        "patients",
                        "patient={}".format(patient),
                    )
                    Folder.create(epoch_patient_dir)

                    # create cross validation dir to save distance maps and clicks
                    if inference_type == "idl.gtvn":
                        cross_valid_patient_dir = os.path.join(
                            Path(fold_dir).parent,
                            "patients",
                            "patient={}".format(patient),
                            "round=01",
                        )
                        Folder.create(cross_valid_patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self._single_patient_inference(
                        inference_type=inference_type,
                        patient=patient,
                        cnn=cnn,
                        dataset_ver=dataset_ver,
                        segment_metrics=segment_metrics,
                        baseline_id=Path(fold_dir).parent.parent.name,
                    )

                    # baseline: save preds of current patient
                    if inference_type == "baseline":
                        for gtv in ["gtvt", "gtvn"]:
                            Nii.save(
                                img=patient_results[gtv]["pred"],
                                save_path=os.path.join(
                                    epoch_patient_dir, "{}_pred.nii".format(gtv)
                                ),
                                spacing=g.NII_SPACING[dataset_ver],
                            )
                    # idl.gtvn: save pred, distance map and clicks of current patient
                    else:
                        # save pred
                        Nii.save(
                            img=patient_results["gtvn"]["pred"],
                            save_path=os.path.join(epoch_patient_dir, "gtvn_pred.nii"),
                            spacing=g.NII_SPACING[dataset_ver],
                        )
                        # save distance map and clicks
                        for i in ["distance.map", "clicks"]:
                            save_path = os.path.join(
                                cross_valid_patient_dir,
                                "gtvn_{}.nii".format(i.replace(".", "_")),
                            )
                            if not os.path.exists(save_path):
                                Nii.save(
                                    img=patient_results["gtvn"][i],
                                    save_path=save_path,
                                    spacing=g.NII_SPACING[dataset_ver],
                                )

                    # record score of current patient (test and valid sets only)
                    if "test" in dataset_section or "valid" in dataset_section:
                        for gtv in patient_results.keys():
                            for metric in g.METRICS:
                                # baseline
                                if inference_type == "baseline":
                                    # save cur patient score
                                    epoch_scores["patient={}".format(patient)][gtv][
                                        metric
                                    ] = patient_results[gtv][metric]
                                    # add scores of current patient into avg and median
                                    for stats in ["median", "avg"]:
                                        epoch_scores[stats][gtv][metric].append(
                                            patient_results[gtv][metric]
                                        )
                                # idl.gtvn
                                else:
                                    # save cur patient score
                                    epoch_scores["patient={}".format(patient)][metric][
                                        "round=01"
                                    ] = patient_results[gtv][metric]
                                    # add scores of current patient into median(list)
                                    for stats in ["median", "avg"]:
                                        epoch_scores[stats][metric]["round=01"].append(
                                            patient_results[gtv][metric]
                                        )

                # all patients under current epoch have been traversed
                # calculate median and avg score of current epoch
                if "test" in dataset_section or "valid" in dataset_section:
                    self.__inference_calculate_avg_mean_then_save(
                        inference_type=inference_type,
                        scores=epoch_scores,
                    )
                    # save scores in json
                    Json.save(
                        data=epoch_scores,
                        path=os.path.join(
                            epoch_dir,
                            "inference_{}_{}.json".format(dataset_ver, dataset_section),
                        ),
                    )
                continue  # next epoch

    def __inference_calculate_avg_mean_then_save(
        self, inference_type: str, scores: Dict
    ):
        # baseline
        if inference_type == "baseline":
            for gtv in ["gtvs", "gtvt", "gtvn"]:
                for metric in g.METRICS:
                    scores["median"][gtv][metric] = Value.median(
                        scores["median"][gtv][metric]
                    )
                    scores["avg"][gtv][metric] = Value.avg(scores["avg"][gtv][metric])
        # idl.gtvn
        else:
            for metric in g.METRICS:
                scores["median"][metric]["round=01"] = Value.median(
                    scores["median"][metric]["round=01"]
                )
                scores["avg"][metric]["round=01"] = Value.avg(
                    scores["avg"][metric]["round=01"]
                )

    def cross_valid_evaluation(
        self,
        baseline_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        self._cross_valid_evaluation(
            inference_type="baseline",
            train_id=baseline_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
            debug_mode=debug_mode,
        )

    def _cross_valid_evaluation(
        self,
        inference_type: str,
        train_id: str,
        dataset_section: str,  # train/test.inter/test.exter/test
        dataset_ver: str = None,  # au.1mm/au.3mm/mda
        debug_mode: bool = False,
    ):
        if not train_id.startswith(inference_type):
            Debug.error_exit("train id error")
        print("")
        print("calculate cross valid scores: {}".format(train_id))

        if "valid" in dataset_section:
            Debug.error_exit(
                "use 'train' instead of 'valid' in cross validation evaluation"
            )

        dataset_ver, fold_dirs, segment_metrics = self.__inference_prepare(
            inference_type=inference_type,
            train_id=train_id,
            dataset_section=dataset_section,
            dataset_ver=dataset_ver,
        )

        # create folder in train_dir to save cross_valid preds
        Folder.create(os.path.join(Path(fold_dirs[0]).parent, "patients"))

        patients = self._load_patients(
            dataset_ver=dataset_ver,
            fold=0,  # fold=0 means no validation set, but put all folds in training set
            debug_mode=debug_mode,
        )

        # initialize scores dict
        scores = self.__inference_init_scores(
            inference_type=inference_type,
            baseline_id=Path(fold_dirs[0]).parent.parent.name,
            dataset_ver=dataset_ver,
            dataset_section=dataset_section,
            patients=patients,
        )

        for patient in tqdm(patients[dataset_section]):
            # initialize preds
            preds = Dict()
            if inference_type == "baseline":
                gtv_list = ["gtvs", "gtvt", "gtvn"]
            else:
                gtv_list = ["gtvn"]
            for gtv in gtv_list:
                preds[gtv] = None

            for fold_dir in fold_dirs:
                # find epoch dir
                epoch_dirs = Directory.get_sub_folders(
                    fold_dir, key_word="epoch=", full_path=True
                )
                if len(epoch_dirs) > 1:
                    self.remove_non_optimal_epochs(train_id)
                    epoch_dir = Directory.get_sub_folders(
                        fold_dir, key_word="epoch=", full_path=True
                    )[0]
                else:
                    epoch_dir = epoch_dirs[0]

                # load preds
                patient_dir = os.path.join(
                    epoch_dir, "patients", "patient={}".format(patient)
                )
                if inference_type == "baseline":
                    gtv_list = ["gtvt", "gtvn"]
                else:
                    gtv_list = ["gtvn"]
                for gtv in gtv_list:
                    img = Nii.load(os.path.join(patient_dir, "{}_pred.nii".format(gtv)))
                    if preds[gtv] is None:
                        preds[gtv] = img
                    else:
                        preds[gtv] += img

            # all folds is walked
            if inference_type == "baseline":
                preds["gtvs"] = preds["gtvt"] + preds["gtvn"]
                gtv_list = ["gtvt", "gtvn"]
            else:
                gtv_list = ["gtvn"]

            for gtv in gtv_list:
                preds[gtv] /= len(fold_dirs)

            # create cross_valid dir
            pred_dir = os.path.join(
                Path(fold_dirs[0]).parent, "patients", "patient={}".format(patient)
            )
            if inference_type == "idl.gtvn":
                pred_dir = os.path.join(pred_dir, "round=01")
            Folder.create(pred_dir)

            # save cross_valid preds
            for gtv in gtv_list:
                Nii.save(
                    img=preds[gtv],
                    save_path=os.path.join(pred_dir, "{}_pred.nii".format(gtv)),
                    spacing=g.NII_SPACING[dataset_ver],
                )

            # load labels and calculate metrics (on test set only)
            if "test" in dataset_section:
                labels = Img.load_labels(
                    dataset_dir=g.DATASET_DIR[dataset_ver], patient=patient
                )
                if inference_type == "baseline":
                    gtv_list = ["gtvs", "gtvt", "gtvn"]
                else:
                    gtv_list = ["gtvn"]

                for gtv in gtv_list:
                    for metric in g.METRICS:
                        score = segment_metrics[metric](preds[gtv], labels[gtv])
                        # record current score
                        if inference_type == "baseline":
                            scores["patient={}".format(patient)][gtv][metric] = score
                        else:
                            scores["patient={}".format(patient)][metric][
                                "round=01"
                            ] = score
                        # record scores for avg and median score calculation
                        for stats in ["median", "avg"]:
                            if inference_type == "baseline":
                                scores[stats][gtv][metric].append(score)
                            else:
                                scores[stats][metric]["round=01"].append(score)

        # all patients have been traversed
        if "test" in dataset_section:
            # calculate avg and median score
            self.__inference_calculate_avg_mean_then_save(
                inference_type=inference_type, scores=scores
            )
            # save scores in json
            Json.save(
                data=scores,
                path=os.path.join(
                    Path(fold_dirs[0]).parent,
                    "inference_{}_{}.json".format(dataset_ver, dataset_section),
                ),
            )

    def remove_non_optimal_epochs(self, baseline_id: str):
        self._remove_non_optimal_epochs(inference_type="baseline", train_id=baseline_id)

    def _remove_non_optimal_epochs(self, inference_type: str, train_id: str):
        if not train_id.startswith(inference_type):
            Debug.error_exit("train id error")
        print("")
        print("remove non optimal epochs: {}".format(train_id))

        train_dir = self._find_train_dir(train_id)
        fold_dirs = Directory.get_sub_folders(
            train_dir, key_word="fold=", full_path=True
        )

        # load dataset version
        dataset_ver = Json.load(os.path.join(fold_dirs[0], "hyper.json"))["dataset.ver"]
        Value.is_valid_dataset_version(dataset_ver=dataset_ver)

        inference_json_name = "inference_{}_valid.json".format(dataset_ver)

        for fold_dir in fold_dirs:
            fold_scores = Dict()

            for epoch_dir in Directory.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name

                # load scores of current epoch
                # if baseline
                epoch_scores = Json.load(os.path.join(epoch_dir, inference_json_name))
                for stats in ["median", "avg"]:
                    if inference_type == "baseline":
                        fold_scores[epoch][stats] = epoch_scores[stats]
                    else:
                        for metric in g.METRICS:
                            fold_scores[epoch][stats][metric] = epoch_scores[stats][
                                metric
                            ]["round=01"]

            best_epoch = self.__find_best_result(
                inference_type=inference_type, scores=fold_scores
            )

            # delete non-optimal epochs
            for epoch_dir in Directory.get_sub_folders(
                fold_dir, key_word="epoch=", full_path=True
            ):
                epoch = Path(epoch_dir).name
                if epoch != best_epoch:
                    Folder.delete(epoch_dir)
                    print("delete: {} {}".format(Path(fold_dir).name, epoch))

    # a sub function of _remove_non_optimal_epochs()
    def __find_best_result(self, inference_type: str, scores: Dict):
        if inference_type == "baseline":
            gtv_list = ["gtvs", "gtvt", "gtvn"]
        else:
            gtv_list = ["gtvn"]

        for stats in ["median", "avg"]:
            for gtv in gtv_list:
                for metric in g.METRICS:
                    # create a tmp list to sort
                    list_to_sort = List()

                    # add elements into the list
                    for epoch in scores.keys():
                        if inference_type == "baseline":
                            cur_score = scores[epoch][stats][gtv][metric]
                        else:
                            cur_score = scores[epoch][stats][metric]
                        list_to_sort.append(cur_score)

                    # sort the list
                    if metric == "dsc":
                        list_to_sort.sort(reverse=False)
                    else:
                        list_to_sort.sort(reverse=True)

                    # update value based on the idx in the list
                    for epoch in scores.keys():
                        if inference_type == "baseline":
                            cur_score = scores[epoch][stats][gtv][metric]
                        else:
                            cur_score = scores[epoch][stats][metric]

                        new_value = list_to_sort.index(cur_score)
                        if metric == "dsc":
                            new_value *= 2

                        if inference_type == "baseline":
                            scores[epoch][stats][gtv][metric] = new_value
                        else:
                            scores[epoch][stats][metric] = new_value

        evaluation = Dict()
        for epoch in scores:
            evaluation[epoch] = 0
            for stats in ["avg", "median"]:
                for gtv in gtv_list:
                    for metric in g.METRICS:
                        if inference_type == "baseline":
                            evaluation[epoch] += scores[epoch][stats][gtv][metric]
                        else:
                            evaluation[epoch] += scores[epoch][stats][metric]
        return evaluation.key_with_max_value()
