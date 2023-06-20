import os
from training_baseline import TrainingBaseline
from loss_func_idl_gtvn import UnifiedFocalLossIDLGTVn
from dataset_idl_gtvn import DataSetIDLGTVn
from custom import Global as g
from custom import Folder
from custom import Explorer
from custom import ValueUtils
from custom import Dict
from pathlib import Path


class TrainingIDLGTVn(TrainingBaseline):
    # if float64 needed, use: "cnn.to(torch.double)"
    def __load_cnn(self, hyper: Dict, cnn_path: str = None):
        # new model
        if cnn_path == "" or cnn_path is None:
            if isinstance(hyper["cnn"], DataParallel):
                hyper["cnn"] = hyper["cnn"].module

            if hyper["cnn"] == "unet.pp.slim" or isinstance(hyper["cnn"], UNetPPSlim):
                if hyper["train.type"] == "baseline":
                    in_chan = 4
                    out_chan = 3
                elif hyper["train.type"] == "idl_gtvn":
                    in_chan = 5
                    out_chan = 2
                elif hyper["train.type"] == "idl":
                    in_chan = 6
                    out_chan = 3

                hyper["cnn"] = UNetPPSlim(
                    in_chan=in_chan, out_chan=out_chan, dropout=hyper["dropout"]
                ).to(g.DEVICE)

            elif hyper["cnn"] == "unet.slim" or isinstance(hyper["cnn"], UNetSlim):
                hyper["cnn"] = UNetSlim(
                    in_chan=5, out_chan=2, edge_chan=16, dropout=hyper["dropout"]
                ).to(g.DEVICE)

        # existing model
        else:
            hyper["cnn"] = torch.load(cnn_path).to(g.DEVICE)

        # set multi-GPU
        if GPU.used_count() > 1:
            hyper["cnn"] = DataParallel(hyper["cnn"]).to(g.DEVICE)

    def __load_unique_hyper(self, hyper: Dict, debug_mode: bool):
        # run this first
        self._load_common_hyper(hyper=hyper, debug_mode=debug_mode)

        # loss function
        hyper["loss.func"] = UnifiedFocalLossIDLGTVn(
            asym=hyper["loss.asym"],
            weight=hyper["loss.weight"],
            delta=hyper["loss.delta"],
            gamma=hyper["loss.gamma"],
        ).to(g.DEVICE)

        # load train/valid/test datasets
        augment = Dict()
        augment["methods"] = hyper["augment.methods"]
        augment["pct"] = hyper["augment.pct"]
        augment["min"] = hyper["augment.min"]
        augment["max"] = hyper["augment.max"]

        for i in ["train", "valid", "test.inter"]:
            hyper["{}.set".format(i)] = DataSetIDLGTVn(
                patients=hyper["{}.patients".format(i)],
                baseline_id=hyper["baseline.id"],
                augment=augment,
                random_click=False,
            )
            augment = None

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
                hyper=hyper, train_result_dir=idl_gtvn_dir, debug_mode=debug_mode
            )

            # inference
            self.inference(idl_gtvn_id=idl_gtvn_id, debug_mode=debug_mode)

    def remove_non_optimal_epochs(self, idl_gtvn_id: str, dataset: str = "valid"):
        self._remove_non_optimal_epochs(train_id=idl_gtvn_id, dataset=dataset)

    def __single_patient_inference(
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

    def inference(self, idl_gtvn_id: str, debug_mode: bool = False):
        print("")
        print("inference: {}".format(idl_gtvn_id))

        # find idl gtvn folder
        idl_gtvn_dir = self._find_result_dir(idl_gtvn_id)
        if idl_gtvn_dir is None:
            print("idl_gtvn_id not found")
            return

        # this is only for idl_gtvn to load baseline gtvn preds
        baseline_epoch_dir = os.path.join(Path(idl_gtvn_dir).parent, "baseline")

        # loop through fold dirs
        for fold_dir in Explorer.get_sub_folders(
            train_result_dir, key_word="fold=", full_path=True
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
                if inference_type == "baseline":
                    cnn_dir = os.path.join(epoch_dir, "baseline")
                else:
                    cnn_dir = epoch_dir
                cnn_path = Explorer.get_sub_files(
                    cnn_dir, key_word=".pt", full_path=True
                )[0]
                hyper = Dict()  # create an empty hyper dict to save cnn
                self._load_cnn(hyper=hyper, cnn_path=cnn_path)

                # load dataset patients
                train_patients, valid_patients, test_patients = self._load_patients(
                    fold=fold, debug_mode=debug_mode
                )
                if dataset == "test.inter":
                    patients = test_patients
                elif dataset == "valid":
                    patients = valid_patients
                elif dataset == "train":
                    patients = train_patients

                # initialize scores dict (only on test and valid set)
                if dataset == "test.inter" or dataset == "valid":
                    epoch_scores = Dict()
                    if inference_type == "baseline":
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                for stats in ["median", "avg"]:
                                    epoch_scores[i][gtv][metric] = List()
                    else:  # "idl"
                        if dataset == "valid":
                            # no need to record baseline score for valid set
                            for metric in g.METRICS:
                                for stats in ["median", "avg"]:
                                    epoch_scores[i][metric] = List()
                        elif dataset == "test.inter":
                            # record baseline score in ["round=00"] for test set
                            # so here, initialize ["round=01"] as a list
                            for metric in g.METRICS:
                                for stats in ["median", "avg"]:
                                    epoch_scores[i][metric]["round=01"] = List()
                            # copy baseline scores of each patient
                            baseline_scores = Json.load(
                                os.path.join(
                                    idl_gtvn_baseline_epoch_dir,
                                    "baseline",
                                    "inference_test.json",
                                )
                            )
                            for patient in patients:
                                for metric in g.METRICS:
                                    epoch_scores["patient={}".format(patient)][metric][
                                        "round=00"
                                    ] = baseline_scores["patient={}".format(patient)][
                                        "gtvn"
                                    ][
                                        metric
                                    ]
                            # also copy median score of each patient
                            for metric in g.METRICS:
                                for stats in ["median", "avg"]:
                                    epoch_scores[i][metric][
                                        "round=00"
                                    ] = baseline_scores[i]["gtvn"][metric]

                for patient in tqdm(patients):
                    # create folder to save cur patient preds and scores
                    if inference_type == "baseline":
                        patient_dir = os.path.join(
                            epoch_dir,
                            "baseline",
                            "patients",
                            "patient={}".format(patient),
                        )
                        Folder.create(patient_dir)

                    # for idl gtvn, only create patient folders for test set
                    if (
                        inference_type == "idl_gtvn" or inference_type == "idl"
                    ) and dataset == "test.inter":
                        patient_dir = os.path.join(
                            epoch_dir,
                            "patients",
                            "patient={}".format(patient),
                        )
                        Folder.create(patient_dir)

                    # results structure: gtvs/gtvt/gtvn: {pred, dsc, msd, hd95}
                    patient_results = self._patient_inference(
                        patient=patient,
                        hyper=hyper,
                        inference_type=inference_type,
                        idl_gtvn_baseline_epoch_dir=idl_gtvn_baseline_epoch_dir,
                    )

                    # save preds of current patient
                    if inference_type == "baseline":
                        # save both gtvt and gtvn, for idl/idl_gtvt/idl_gtvn
                        gtv_list = ["gtvt", "gtvn"]
                    else:
                        if dataset == "test.inter":
                            gtv_list = ["gtvn"]
                            # save clicks.nii
                            Nii.save(
                                img=patient_results["gtvn"]["distance.map"],
                                save_path=os.path.join(patient_dir, "distance_map.nii"),
                                spacing=g.NII_SPACING,
                            )
                            Nii.save(
                                img=patient_results["gtvn"]["clicks"],
                                save_path=os.path.join(patient_dir, "clicks.nii"),
                                spacing=g.NII_SPACING,
                            )
                        else:
                            gtv_list = []

                    for gtv in gtv_list:
                        Nii.save(
                            img=patient_results[gtv]["pred"],
                            save_path=os.path.join(
                                patient_dir, "{}_pred.nii".format(gtv)
                            ),
                            spacing=g.NII_SPACING,
                        )

                    # record score of current patient
                    if inference_type == "baseline":
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                # save cur patient score into inference_test.json (test set only)
                                if dataset == "test.inter":
                                    epoch_scores["patient={}".format(patient)][gtv][
                                        metric
                                    ] = patient_results[gtv][metric]
                                # add scores of current patient into median(list)
                                if dataset == "test.inter" or dataset == "valid":
                                    for stats in ["median", "avg"]:
                                        epoch_scores[i][gtv][metric].append(
                                            patient_results[gtv][metric]
                                        )
                    else:
                        for metric in g.METRICS:
                            if dataset == "test.inter":
                                # save cur patient score into inference_test.json (test set only)
                                epoch_scores["patient={}".format(patient)][metric][
                                    "round=01"
                                ] = patient_results["gtvn"][metric]
                                # add scores of current patient into median(list)
                                # record in ["round=01"] for test set
                                for stats in ["median", "avg"]:
                                    epoch_scores[i][metric]["round=01"].append(
                                        patient_results["gtvn"][metric]
                                    )
                            # add scores of current patient into median(list)
                            if dataset == "valid":
                                for stats in ["median", "avg"]:
                                    epoch_scores[i][metric].append(
                                        patient_results["gtvn"][metric]
                                    )

                # all patients under current epoch have been traversed
                # no need to calculate median score on training set
                if dataset == "train":
                    continue  # next epoch dir

                if dataset == "test.inter" or dataset == "valid":
                    # calculate median score (test and valid set only)
                    if inference_type == "baseline":
                        for gtv in ["gtvs", "gtvt", "gtvn"]:
                            for metric in g.METRICS:
                                epoch_scores["median"][gtv][metric] = statistics.median(
                                    epoch_scores["median"][gtv][metric]
                                )
                                epoch_scores["avg"][gtv][metric] = ValueUtils.avg(
                                    epoch_scores["avg"][gtv][metric]
                                )
                    else:  # idl/idl_gtvt/idl_gtvn
                        for metric in g.METRICS:
                            if dataset == "test.inter":
                                epoch_scores["median"][metric][
                                    "round=01"
                                ] = statistics.median(
                                    epoch_scores["median"][metric]["round=01"]
                                )
                                epoch_scores["avg"][metric][
                                    "round=01"
                                ] = ValueUtils.avg(
                                    epoch_scores["avg"][metric]["round=01"]
                                )
                            elif dataset == "valid":
                                epoch_scores["median"][metric] = statistics.median(
                                    epoch_scores["median"][metric]
                                )
                                epoch_scores["avg"][metric] = ValueUtils.avg(
                                    epoch_scores["avg"][metric]
                                )
                    # save all patients scores in "inference_test.json"
                    if inference_type == "baseline":
                        json_save_path = os.path.join(
                            epoch_dir, "baseline", "inference_{}.json".format(dataset)
                        )
                    else:
                        json_save_path = os.path.join(
                            epoch_dir, "inference_{}.json".format(dataset)
                        )
                    Json.save(
                        data=epoch_scores,
                        path=json_save_path,
                    )
                    continue  # next epoch dir
