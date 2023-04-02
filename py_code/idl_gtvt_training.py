import os
import random
import statistics
from datetime import datetime
from pathlib import Path
from loss_func import UnifiedFocalLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import global_elems as g
from idl_gtvt_dataset import IDLGTVtDataSet
from training import Training
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import measurements
from custom import Dict
from custom import List
from custom import Json
from custom import set_range


class IDLGTVtTraining(Training):
    def __load_next_round_lr(self, next_round: int, hyper: Dict):
        # hyper["lr"]["init"] is a list of lr of each round
        if next_round > len(hyper["lr"]["init"]):
            next_round = len(hyper["lr"]["init"])

        if g.used_gpu_count() > 1:
            hyper["lr"]["actual"].append(
                hyper["lr"]["init"][next_round - 1] * g.used_gpu_count()
            )
        else:
            hyper["lr"]["actual"].append(hyper["lr"]["init"][next_round - 1])

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr"]["actual"][-1]
        )

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        hyper["scheduler"] = ReduceLROnPlateau(
            optimizer=hyper["optim"],
            mode="min",
            factor=hyper["lr"]["decay.factor"],  # "factor=1" will cause an error
            patience=hyper["lr"]["decay.patience"],
            min_lr=hyper["lr"]["min"],
        )

    # reset cnn/optimizer/scheduler before next patient
    def __reset_cnn(self, hyper: dict, baseline_cnn_path: str):
        # reload cnn
        super()._load_cnn(hyper=hyper, cnn_path=baseline_cnn_path)

        # optimizer (no need to move to cuda)
        hyper["optim"] = optim.Adam(
            params=hyper["cnn"].parameters(), lr=hyper["lr"]["actual"][0]
        )

        # scheduler
        # (1) mode = min(default): lr will reduce when the watched parameter stops decreasing
        # (2) mode = max: lr will reduce when the watched parameter stops increasing
        # (3) factor: new_lr = lr * factor
        # (4) patience: lr will reduce after how many epochs
        hyper["scheduler"] = ReduceLROnPlateau(
            optimizer=hyper["optim"],
            mode="min",
            factor=hyper["lr"]["decay.factor"],  # "factor=1" will cause an error
            patience=hyper["lr"]["decay.patience"],
            min_lr=hyper["lr"]["min"],
        )

    def _load_hyper(
        self, hyper: Dict, baseline_cnn_path: str, debug_mode: bool = False
    ):
        # iter
        if debug_mode:
            # at least 2 iters to compare loss difference
            hyper["iter"] = 2
        else:
            hyper["iter"] = set_range(hyper["iter"], (1, None))

        # lr
        # lr is saved in json file as a string, not a list, because:
        # (1) string is easier to read than list in json file (only one line)
        # (2) a "list" will be recognized as multiple trainings
        hyper["lr"]["init"] = List(hyper["lr"]["init"])
        for i in range(len(hyper["lr"]["init"])):
            hyper["lr"]["init"][i] = float(hyper["lr"]["init"][i])
            hyper["lr"]["init"][i] = set_range(hyper["lr"]["init"][i], (g.EPS, 1))
            # check min lr, make sure it is lower than any lr in the lr list
            hyper["lr"]["min"] = set_range(
                hyper["lr"]["min"], (g.EPS, hyper["lr"]["init"][i])
            )

        # actual lr
        hyper["lr"]["actual"] = List()
        if g.used_gpu_count() > 1:
            hyper["lr"]["actual"].append(hyper["lr"]["init"][0] * g.used_gpu_count())
        else:
            hyper["lr"]["actual"].append(hyper["lr"]["init"][0])

        # lr decay patience (before shared hyper)
        hyper["lr"]["decay.patience"] = set_range(
            hyper["lr"]["decay.patience"], (1, hyper["iter"])
        )

        # augmentation times
        hyper["augment"]["times"] = set_range(hyper["augment"]["times"], (1, None))

        # augmentation percent (based on augment_times)
        hyper["augment"]["pct"] = hyper["augment"]["times"] / (
            hyper["augment"]["times"] + 1
        )

        # select step
        # select.step is saved in json file as a string, not a list, because:
        # (1) string is easier to read than list in json file (only one line)
        # (2) a "list" will be recognized as multiple trainings
        for plane in ["transverse", "coronal", "sagittal"]:
            hyper["select.step"][plane] = List(hyper["select.step"][plane])
            for i in range(len(hyper["select.step"][plane])):
                hyper["select.step"][plane][i] = int(hyper["select.step"][plane][i])
                hyper["select.step"][plane][i] = set_range(
                    hyper["select.step"][plane][i], (0, None)
                )

        # select scenario
        for plane in ["transverse", "coronal", "sagittal"]:
            if (
                hyper["select.scenario"][plane] != "largest"
                and hyper["select.scenario"][plane] != "gravity.center"
                and hyper["select.scenario"][plane] != "equal.divide"
            ):
                hyper["select.scenario"][plane] = "random"

        # weight map parameters
        hyper["weight"]["background"] = set_range(
            hyper["weight"]["background"], (0.0, 1.0)
        )
        hyper["weight"]["slice"] = set_range(
            hyper["weight"]["slice"], (hyper["weight"]["background"], None)
        )
        hyper["weight"]["fp.fn"] = set_range(
            hyper["weight"]["fp.fn"], (hyper["weight"]["slice"], None)
        )
        hyper["weight"]["distance.step"] = set_range(
            hyper["weight"]["distance.step"], (1, None)
        )
        hyper["weight"]["prev.round.decay"] = set_range(
            hyper["weight"]["prev.round.decay"], (0.0, 1.0)
        )

        # load patients
        hyper["patients"] = self.__load_dataset(debug_mode)

        # load shared hyper
        super()._load_hyper(hyper=hyper, cnn_path=baseline_cnn_path)

        # run this after shared hyper loaded, because loss parameters are needed
        hyper["loss"]["func"] = UnifiedFocalLoss(
            asym=hyper["loss"]["asym"],
            weight=hyper["loss"]["weight"],
            delta=hyper["loss"]["delta"],
            gamma=hyper["loss"]["gamma"],
            gtvt_only=True,
        ).to(g.DEVICE)

    def __load_dataset(self, debug_mode: bool = False):
        json_data = Json.load(g.DATASET_SPLIT_JSON)
        test_patients = List(json_data["test.set"])

        # debug mode, only 1 or 2 patients
        if debug_mode:
            test_patients = test_patients[:2]

        return test_patients

    def __get_simple_hyper(self, hyper: Dict) -> Dict:
        simple_hyper = Dict()
        for i in hyper:
            if i == "lr":
                simple_hyper[i] = hyper[i].copy()
                for k in ["init", "actual"]:
                    simple_hyper[i][k] = simple_hyper[i][k].to_str()

            elif i == "patients":
                simple_hyper[i] = len(hyper[i])

            elif i == "select.step":
                simple_hyper[i] = hyper[i].copy()
                for plane in simple_hyper[i]:
                    simple_hyper[i][plane] = simple_hyper[i][plane].to_str()

            elif isinstance(hyper[i], list) or isinstance(hyper[i], dict):
                simple_hyper[i] = hyper[i].copy()
            else:
                simple_hyper[i] = hyper[i]

        return simple_hyper

    def __print_hyper(self, hyper: Dict):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._print_hyper(simple_hyper)

    def __save_hyper(self, hyper: Dict, json_path: str):
        simple_hyper = self.__get_simple_hyper(hyper)
        super()._save_hyper(simple_hyper, json_path)

    # def real_training(
    #     self,
    #     baseline_id: str,
    #     idl_results_folder: str,
    #     idl_gtvt_id: str,
    #     cur_patient: str,
    #     cur_round: int,
    #     debug_mode: bool = False,
    # ):
    #     self._idl_gtvt_id = idl_gtvt_id

    #     # get baseline cnn and hyper path
    #     baseline_cnn_path, baseline_hyper_path = self.__get_baseline_paths(baseline_id)
    #     g.print_line()
    #     print(baseline_cnn_path)
    #     # load hypers
    #     idl_hyper_dict = Json.load(g.HYPER_JSON_IDL_GTVT)
    #     baseline_hyper_dict = Json.load(baseline_hyper_path)

    #     # make sure all hypers are unique, no arrangement
    #     hyper = Dict()
    #     for i in idl_hyper_dict:
    #         if isinstance(idl_hyper_dict[i], list):
    #             hyper[i] = idl_hyper_dict[i][0]
    #         else:
    #             hyper[i] = idl_hyper_dict[i]

    #     # load and print hyper
    #     self._load_hyper(
    #         hyper=hyper,
    #         baseline_cnn_path=baseline_cnn_path,
    #         debug_mode=debug_mode,
    #     )
    #     self.__print_hyper(hyper)

    #     # check if result folder exist
    #     cur_result_folder = os.path.join(idl_results_folder, self._idl_id)
    #     if not os.path.exists(cur_result_folder):
    #         g.exit_app("IDLGTVtTraining.real_training(): iDL result folder doesn't exist")

    #     # create json file to save train loss
    #     train_loss_dict = Dict()
    #     train_loss_dict["iter"] = Dict()
    #     Json.save(
    #         train_loss_dict,
    #         os.path.join(
    #             cur_result_folder, "patient={}".format(cur_patient), "train_loss.json"
    #         ),
    #     )

    #     # get annotated slices
    #     cur_round_folder = os.path.join(
    #         cur_result_folder,
    #         "patient={}".format(cur_patient),
    #         "round={:02d}".format(cur_round),
    #     )
    #     annotated_slices = Dict()
    #     annotated_slices["round=01"] = List()  # doesn't matter what the dict key is
    #     for file_name in g.get_sub_files(cur_round_folder, key_word="_label.npy"):
    #         slice_id = file_name[len("slice_") : -len("_label.npy")]
    #         slice_id = slice_id.zfill(3)
    #         annotated_slices["round=01"].append(slice_id)

    #     # training start time
    #     hyper["time.spent"] = datetime.now()

    #     self.__training_cur_round(
    #         cur_result_folder=cur_result_folder,
    #         cur_patient=cur_patient,
    #         annotated_slices=annotated_slices,
    #         label_folder=cur_round_folder,
    #     )

    #     # get training time spent before save hyper
    #     hyper["time.spent"] = datetime.now() - hyper["time.spent"]
    #     # save hyper
    #     self.__save_hyper(os.path.join(cur_result_folder, "hyper.json"))

    # in this function, cur round slices have not been added into annotated_slices
    def __select_cur_round_slices(
        self,
        annotated_slices: Dict,
        hyper: Dict,
        patient_folder: str,
    ) -> list:  # return a list of int

        cur_round_slices = Dict()
        for plane in ["transverse", "coronal", "sagittal"]:
            cur_round_slices[plane] = List()

        cur_round = max(
            len(annotated_slices["transverse"]),
            len(annotated_slices["coronal"]),
            len(annotated_slices["sagittal"]),
        )
        cur_round += 1

        patient = Path(patient_folder).name
        patient = patient[len("patient=") :]

        label = g.load_nii(
            os.path.join(g.DATASET_FOLDER, "HNCDL_{}_GTVt.nii".format(patient)),
            binary=True,
        )
        label_center = measurements.center_of_mass(label)

        # select slices through each plane
        for plane in ["transverse", "coronal", "sagittal"]:

            # skip cur plane if no slice needs to be selected
            if len(hyper["select.step"][plane]) < cur_round:
                continue

            candidate_slices = Dict()
            cur_plane_annotated_slices = g.dict_to_list(annotated_slices[plane])

            # go through pred and record tumor size
            if plane == "transverse":
                slice_counts = label.shape[0]
            elif plane == "coronal":
                slice_counts = label.shape[1]
            elif plane == "sagittal":
                slice_counts = label.shape[2]

            for cur_slice in range(slice_counts):
                # skip slice that already been annotated
                if cur_slice in cur_plane_annotated_slices:
                    continue
                else:
                    if plane == "transverse":
                        cur_slice_tumor_size = label[cur_slice, :, :].sum()
                    elif plane == "coronal":
                        cur_slice_tumor_size = label[:, cur_slice, :].sum()
                    elif plane == "sagittal":
                        cur_slice_tumor_size = label[:, :, cur_slice].sum()
                    # add slice with target (pred or label) into candidates
                    if cur_slice_tumor_size > 0:
                        candidate_slices[cur_slice] = cur_slice_tumor_size

            # "largest"
            if hyper["select.scenario"][plane] == "largest":
                # descrease sort the dict (return a list of tuple)
                candidate_slices = g.sort_dict_by_value(candidate_slices, reverse=True)
                cur_round_slices[plane] = candidate_slices.keys()

            # "gravity.center", round =1
            elif hyper["select.scenario"][plane] == "gravity.center" and cur_round == 1:
                if plane == "transverse":
                    cur_round_slices[plane].append(round(label_center[0]))
                elif plane == "coronal":
                    cur_round_slices[plane].append(round(label_center[1]))
                elif plane == "sagittal":
                    cur_round_slices[plane].append(round(label_center[2]))

            # "equal.divide", round = 1
            elif hyper["select.scenario"][plane] == "equal.divide" and cur_round == 1:
                divided_parts = hyper["select.step"][plane][0] + 1
                candidate_slices = candidate_slices.keys()
                for part in range(1, divided_parts):
                    idx = len(candidate_slices) * part / divided_parts
                    idx = round(idx)
                    idx = set_range(idx, (1, len(candidate_slices)))
                    cur_round_slices[plane].append(candidate_slices[idx - 1])

            # (1) "random"
            # (2) "gravity.center", round >= 2
            # (3) "equal.divide", round >= 2
            else:
                cur_round_slices[plane] = candidate_slices.keys()
                random.shuffle(cur_round_slices[plane])

            # narrow cur_round_slices based on select.step
            if hyper["select.scenario"][plane] == "gravity.center" and cur_round == 1:
                cur_round_slices_count = 1
            else:
                cur_round_slices_count = hyper["select.step"][plane][cur_round - 1]
            if cur_round_slices_count < len(cur_round_slices[plane]):
                cur_round_slices[plane] = cur_round_slices[plane][
                    :cur_round_slices_count
                ]

            # add cur_round_slices into annotated_slices
            annotated_slices[plane][
                "round={:02d}".format(cur_round)
            ] = cur_round_slices[plane]

        return cur_round_slices

    def __inference_cur_round(self, cur_round_folder: str, hyper: Dict):
        cur_round = Path(cur_round_folder).name

        patient = Path(cur_round_folder).parent.name

        # result structure: gtvt: {pred, dsc, msd, hd95}
        patient_result = self._inference_single_patient(
            patient=patient[len("patient=") :], hyper=hyper, gtvt_only=True
        )

        # save score of cur patient
        idl_gtvt_folder = Path(cur_round_folder).parent.parent.parent
        score_json_path = os.path.join(idl_gtvt_folder, "score.json")
        score = Json.load(score_json_path)
        for metric in g.METRICS:
            score[patient][metric][cur_round] = patient_result["gtvt"][metric]
        Json.save(score, score_json_path)

        # save pred of cur patient
        g.save_nii(
            img=patient_result["gtvt"]["pred"],
            save_path=os.path.join(cur_round_folder, "pred_gtvt.nii"),
            spacing=g.NII_SPACING,
        )

    def __training_cur_round(
        self,
        cur_round_folder: str,
        epoch_folder: str,
        label_folder: str,
        hyper: Dict,
        annotated_slices: Dict,
    ):
        g.create_folder(cur_round_folder)

        cur_round = Path(cur_round_folder).name
        cur_round = int(cur_round[len("round=") :])

        patient = Path(cur_round_folder).parent.name
        patient = patient[len("patient=") :]

        idl_gtvt_folder = Path(cur_round_folder).parent.parent.parent
        loss_json_path = os.path.join(Path(cur_round_folder).parent, "loss.json")
        loss_dict = Json.load(loss_json_path)

        if cur_round == 1:
            pred_folder = os.path.join(
                epoch_folder, "baseline", "patients", "patient={}".format(patient)
            )
        else:
            pred_folder = os.path.join(
                Path(cur_round_folder).parent, "round={:02d}".format(cur_round - 1)
            )

        # record current round time spent
        cur_round_time_spent = datetime.now()

        # create iDL dataset
        idl_gtvt_dataset = IDLGTVtDataSet(
            patient=patient,
            annotated_slices=annotated_slices,
            label_folder=label_folder,
            pred_folder=pred_folder,
            augment=hyper["augment"],
            weight=hyper["weight"],
        )

        # optimize batch size (before create dataloader)
        self._optimize_batch_size(hyper=hyper, dataset=idl_gtvt_dataset)

        # idl gtvt dataloader
        idl_gtvt_loader = DataLoader(
            dataset=idl_gtvt_dataset,
            batch_size=hyper["batch.size"],
            shuffle=True,
            num_workers=g.NUM_WORKERS,
        )

        # iter loop
        for cur_iter in tqdm(range(hyper["iter"])):
            hyper["cnn"].train()
            sum_loss = 0
            batch_num = 0

            # freeze layers before iDL
            if hyper["layer.freezing"]:
                if g.used_gpu_count() > 1:
                    # here, hyper["cnn"] is DataParallel, not network itself
                    hyper["cnn"].module.freeze_top()
                else:
                    hyper["cnn"].freeze_top()

            for inputs, labels, weight_map in idl_gtvt_loader:
                # zero grad at the begining of each mini-batch
                hyper["optim"].zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                weight_map = weight_map.to(g.DEVICE)
                outputs = hyper["cnn"](inputs)[3]
                loss = hyper["loss"]["func"](outputs, labels, weight_map)
                loss.backward()  # get grad (must after: optim.zero_grad())
                hyper["optim"].step()  # update param
                sum_loss += loss.item()
                batch_num += 1

            # cur iter finished
            # update scheduler
            iter_loss = sum_loss / batch_num
            hyper["scheduler"].step(iter_loss)

            # record loss
            loss_dict[
                "iter={:03d}".format((cur_round - 1) * hyper["iter"] + (cur_iter + 1))
            ] = iter_loss
            # save loss and update loss figure after every iter, if there is only one patient
            patient_folder_list = g.get_sub_folders(
                os.path.join(idl_gtvt_folder, "patients")
            )
            if len(patient_folder_list) <= 1:
                Json.save(loss_dict, loss_json_path)
                self.__draw_loss_fig(idl_gtvt_folder)

        # current round idl finished
        cnn_save_path = os.path.join(
            cur_round_folder, Path(cur_round_folder).name + ".pt"
        )
        self._save_cnn(hyper, cnn_save_path)

        # inference
        self.__inference_cur_round(cur_round_folder=cur_round_folder, hyper=hyper)

        # save time spent
        cur_round_time_spent = datetime.now() - cur_round_time_spent
        round_str = "round={:02d}".format(cur_round)
        if hyper["time.spent"]["avg"][round_str] == {}:
            hyper["time.spent"]["avg"][round_str] = cur_round_time_spent
        else:
            hyper["time.spent"]["avg"][round_str] += cur_round_time_spent

        # save loss
        Json.save(loss_dict, loss_json_path)

    def __training_cur_patient(
        self,
        patient: str,
        epoch_folder: str,
        idl_gtvt_folder: str,
        hyper: Dict,
    ):
        # create current patient folder
        patient_folder = os.path.join(
            idl_gtvt_folder, "patients", "patient={}".format(patient)
        )
        g.create_folder(patient_folder)
        # create an empty loss.json
        Json.save(Dict(), os.path.join(patient_folder, "loss.json"))

        # initialize idl score (copy from baseline)
        baseline_score = Json.load(
            os.path.join(epoch_folder, "baseline", "score_test.json")
        )
        idl_gtvt_score_path = os.path.join(idl_gtvt_folder, "score.json")
        idl_gtvt_score = Json.load(idl_gtvt_score_path)
        for metric in g.METRICS:
            idl_gtvt_score["patient={}".format(patient)][metric][
                "round=00"
            ] = baseline_score["patient={}".format(patient)]["gtvt"][metric]
        Json.save(idl_gtvt_score, idl_gtvt_score_path)

        g.print_line()
        print("patient:", patient)

        annotated_slices = Dict()

        # loop through each round
        max_round = max(
            len(hyper["select.step"]["transverse"]),
            len(hyper["select.step"]["coronal"]),
            len(hyper["select.step"]["sagittal"]),
        )
        for cur_round in range(1, max_round + 1):

            # cur round slices are add into annotated_slices in this function
            cur_round_slices = self.__select_cur_round_slices(
                annotated_slices=annotated_slices,
                hyper=hyper,
                patient_folder=patient_folder,
            )

            # no slice needs to be annotated in cur round
            if (
                len(cur_round_slices["transverse"]) == 0
                and len(cur_round_slices["coronal"]) == 0
                and len(cur_round_slices["sagittal"]) == 0
            ):
                break

            # start current round
            print("round:", cur_round)

            cur_round_folder = os.path.join(
                patient_folder, "round={:02d}".format(cur_round)
            )
            self.__training_cur_round(
                cur_round_folder=cur_round_folder,
                epoch_folder=epoch_folder,
                label_folder=g.DATASET_FOLDER,
                hyper=hyper,
                annotated_slices=annotated_slices,
            )

            if cur_round == max_round:
                break

            # load new lr before next round
            if hyper["lr"]["reset"]:
                self.__load_next_round_lr(cur_round + 1, hyper)

        # draw avg loss of all trained patients
        self.__draw_loss_fig(idl_gtvt_folder)

        # save annotated slices in cur patient folder
        for plane in ["transverse", "coronal", "sagittal"]:
            for cur_round in annotated_slices[plane]:
                annotated_slices[plane][cur_round] = annotated_slices[plane][
                    cur_round
                ].to_str()

        Json.save(
            data=annotated_slices,
            path=os.path.join(
                idl_gtvt_folder,
                "patients",
                "patient={}".format(patient),
                "annotated_slices.json",
            ),
        )

    def draw_loss_fig(self, idl_gtvt_id: str):
        for i in g.walk_sub_folders(g.TRAIN_RESULTS_FOLDER, key_word=idl_gtvt_id):
            # remove "/" if str endswith it
            if i.endswith("/"):
                i = i[:-1]
            if i.endswith(idl_gtvt_id):
                idl_gtvt_folder = i
                break
        self.__draw_loss_fig(idl_gtvt_folder)

    def __draw_loss_fig(self, idl_gtvt_folder: str):
        # avg loss dict
        avg_loss = Dict()
        for cur_patient_folder in g.get_sub_folders(
            os.path.join(idl_gtvt_folder, "patients"), return_full_path=True
        ):
            cur_patient_loss = Json.load(os.path.join(cur_patient_folder, "loss.json"))
            if avg_loss == {}:
                for i in cur_patient_loss:
                    avg_loss[i] = [cur_patient_loss[i]]
            else:
                for i in avg_loss:
                    avg_loss[i].append(cur_patient_loss[i])

        for i in avg_loss:
            avg_loss[i] = g.get_avg_value(avg_loss[i])

        avg_loss = g.dict_to_list(avg_loss)

        # draw figure
        plt.figure().clear()
        plt.plot(range(1, len(avg_loss) + 1), avg_loss, label="loss")
        plt.legend()
        plt.savefig(os.path.join(idl_gtvt_folder, "loss.png"))

    def simulation(
        self,
        baseline_id: str,
        fold: int = 0,
        epoch: int = 0,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        for hyper in self._load_group_hyper(g.HYPER_JSON_IDL_GTVT):

            idl_gtvt_id = "idl_gtvt_" + self._init_train_id(
                train_remark=train_remark,
                debug_mode=debug_mode,
                hyper_json_path=g.HYPER_JSON_IDL_GTVT,
                hyper=hyper,
            )
            g.print_line()
            print(idl_gtvt_id)

            # find fold folder
            if fold <= 0:
                key_word = "fold="
            else:
                key_word = "fold={:02d}".format(fold)
            fold_folder = g.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id),
                key_word=key_word,
                return_full_path=True,
            )[0]

            # find epoch folder
            if epoch <= 0:
                key_word = "epoch="
            else:
                key_word = "epoch={:03d}".format(epoch)
            epoch_folder = g.get_sub_folders(
                fold_folder, key_word=key_word, return_full_path=True
            )[0]
            baseline_cnn_path = g.get_sub_files(
                os.path.join(epoch_folder, "baseline"),
                key_word=".pt",
                return_full_path=True,
            )[0]

            # load and print hyper
            self._load_hyper(
                hyper=hyper, baseline_cnn_path=baseline_cnn_path, debug_mode=debug_mode
            )
            g.print_line()
            self.__print_hyper(hyper)

            # create idl result folder
            idl_gtvt_folder = os.path.join(epoch_folder, "idl_gtvt", idl_gtvt_id)
            g.create_folder(idl_gtvt_folder)

            # save hyper before training
            hyper_save_path = os.path.join(idl_gtvt_folder, "hyper.json")
            self.__save_hyper(hyper, hyper_save_path)

            # create an empty score json files
            Json.save(Dict(), os.path.join(idl_gtvt_folder, "score.json"))

            # training start time
            hyper["time.spent"]["total"] = datetime.now()

            # patient loop
            for patient in hyper["patients"]:
                self.__training_cur_patient(
                    patient=patient,
                    hyper=hyper,
                    epoch_folder=epoch_folder,
                    idl_gtvt_folder=idl_gtvt_folder,
                )

                # reset cnn/optimizer/scheduler before next patient
                if patient != hyper["patients"][-1]:
                    self.__reset_cnn(hyper=hyper, baseline_cnn_path=baseline_cnn_path)

            # record total time spent
            hyper["time.spent"]["total"] = datetime.now() - hyper["time.spent"]["total"]
            hyper["time.spent"]["total"] = str(hyper["time.spent"]["total"]).split(
                ".", 2
            )[0]

            # record avg time spent per patient
            for cur_round in hyper["time.spent"]["avg"]:
                hyper["time.spent"]["avg"][cur_round] /= len(hyper["patients"])
                hyper["time.spent"]["avg"][cur_round] = str(
                    hyper["time.spent"]["avg"][cur_round]
                ).split(".", 2)[0]

            self.__save_hyper(hyper, hyper_save_path)

            self.__calculate_median_score(idl_gtvt_folder)

    def __calculate_median_score(self, idl_gtvt_folder: str):
        score_json_path = os.path.join(idl_gtvt_folder, "score.json")
        score = Json.load(score_json_path)
        median = Dict()

        # add all patients score in to a list
        for patient in score:
            for metric in g.METRICS:
                for cur_round in score[patient][metric]:
                    if median[metric][cur_round] == {}:
                        median[metric][cur_round] = List()
                    median[metric][cur_round].append(score[patient][metric][cur_round])

        # calculate median score
        for metric in g.METRICS:
            for cur_round in median[metric]:
                score["median"][metric][cur_round] = statistics.median(
                    median[metric][cur_round]
                )
        Json.save(data=score, path=os.path.join(score_json_path))

    def inference(self, idl_gtvt_id: str):
        g.print_line()
        print("inference: {}".format(idl_gtvt_id))

        # find idl gtvt folder
        for i in g.walk_sub_folders(g.TRAIN_RESULTS_FOLDER, key_word=idl_gtvt_id):
            # remove "/" if str endswith it
            if i.endswith("/"):
                i = i[:-1]
            if i.endswith(idl_gtvt_id):
                idl_gtvt_folder = i
                break

        # loop through patients folder
        patient_list = g.get_sub_folders(
            os.path.join(idl_gtvt_folder, "patients"),
            key_word="patient=",
        )
        for cur_patient in tqdm(patient_list):
            cur_patient_folder = os.path.join(idl_gtvt_folder, "patients", cur_patient)

            # loop through each round
            for cur_round_folder in g.get_sub_folders(
                cur_patient_folder, key_word="round=", return_full_path=True
            ):
                # load current round cnn
                cur_round_cnn_path = g.get_sub_files(
                    cur_round_folder, key_word=".pt", return_full_path=True
                )
                cur_round_cnn_path = cur_round_cnn_path[0]
                hyper = Dict()
                self._load_cnn(hyper=hyper, cnn_path=cur_round_cnn_path)

                self.__inference_cur_round(
                    cur_round_folder=cur_round_folder, hyper=hyper
                )

        self.__calculate_median_score(idl_gtvt_folder)
