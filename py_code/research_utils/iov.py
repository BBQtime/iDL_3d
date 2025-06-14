import csv
import math
import os
from pathlib import Path

import global_utils.global_core as g
import matplotlib

# Prevent matplotlib.pyplot from using a GUI (like X11) for rendering.
# Without this line, using breakpoints under X11 without VCXSRV can cause the debugger to freeze.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetPart, DatasetVer, Metric, Stats
from metric_utils.added_path_len import APL
from metric_utils.metric_func import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
    surface_dice,
)
from research_utils.research_core import (
    FONT_SIZE,
    explain_metric,
    get_obs_study_patients_list,
)
from tqdm import tqdm


def calculate_iov(obs_study_id_1: str, obs_study_id_2: str):
    print(f"{obs_study_id_1} vs {obs_study_id_2}")
    obs_study_id = Dict()
    obs_study_id["1"] = obs_study_id_1
    obs_study_id["2"] = obs_study_id_2

    if obs_study_id["1"] == obs_study_id["2"]:
        g.error_exit("2 input obs_study_ids cannot be identical.")

    if obs_study_id["1"].startswith("idl.gtvn_") and obs_study_id["2"] == "label":
        obs_study_id["2"] = obs_study_id["2"].capitalize()
        gtv = "gtvn"
    elif obs_study_id["1"].startswith("idl.gtvt_") and obs_study_id["2"] == "label":
        obs_study_id["2"] = obs_study_id["2"].capitalize()
        gtv = "gtvt"
    elif obs_study_id["1"] == "label" and obs_study_id["2"].startswith("idl.gtvn_"):
        obs_study_id["1"] = obs_study_id["1"].capitalize()
        gtv = "gtvn"
    elif obs_study_id["1"] == "label" and obs_study_id["2"].startswith("idl.gtvt_"):
        obs_study_id["1"] = obs_study_id["1"].capitalize()
        gtv = "gtvt"

    elif obs_study_id["1"].startswith("idl.gtvn_") and obs_study_id["2"].startswith(
        "idl.gtvn_"
    ):
        gtv = "gtvn"
    elif obs_study_id["1"].startswith("idl.gtvt_") and obs_study_id["2"].startswith(
        "idl.gtvt_"
    ):
        gtv = "gtvt"
    else:
        g.error_exit("obs study train id error")

    # init observers
    observer = Dict()
    # get observer from obs study id
    for name in ["Jesper", "Kenneth", "Hanna", "Label"]:
        for idx in ["1", "2"]:
            if name in obs_study_id[idx]:
                observer[idx] = name
    if observer["1"] == observer["2"]:
        g.error_exit("2 observers cannot be identical.")

    # init patients
    patients_list = get_obs_study_patients_list()
    if gtv == "gtvn":
        # patient 536 doesnt have gtvn
        patients_list.remove("536")

    # init metrics dict
    metrics_dict = Dict()

    for result_type in ["idl", "correct"]:
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            metrics_dict[result_type][Stats.AVG][metric] = []

        for patient in tqdm(patients_list):
            patient = "patient={}".format(patient)

            img_data = Dict()
            for idx in ["1", "2"]:
                if observer[idx] == "Label":
                    img_data[idx] = g.load_nii(
                        os.path.join(
                            g.DATASET_DIR[DatasetVer.OBS_STUDY],
                            "HNCDL_{}_GTV{}.nii".format(
                                patient[len("patient=") :], gtv[-1]
                            ),
                        ),
                        binary=True,
                    )
                else:
                    idl_img_dir = os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        "baseline_obs.study",
                        obs_study_id[idx],
                        "patients",
                        patient,
                        "round=01",
                    )
                    if os.path.exists(idl_img_dir):
                        origin_pred = g.load_nii(
                            os.path.join(idl_img_dir, "{}_pred.nii.gz".format(gtv)),
                            binary=True,
                        )
                        if result_type == "idl":
                            img_data[idx] = origin_pred
                        elif result_type == "correct":
                            correction_mask = g.load_nii(
                                os.path.join(
                                    idl_img_dir,
                                    "{}_correction_mask.nii.gz".format(gtv),
                                ),
                                binary=True,
                            )
                            correction = g.load_nii(
                                os.path.join(
                                    idl_img_dir, "{}_correction.nii.gz".format(gtv)
                                ),
                                binary=True,
                            )
                            img_data[idx] = g.combine_pred_correction(
                                origin_pred=origin_pred,
                                correction=correction,
                                correction_mask=correction_mask,
                            )
                    else:
                        img_data[idx] = None

            # both imgs are not empty
            if img_data["1"] is not None and img_data["2"] is not None:
                if img_data["1"].shape != img_data["2"].shape:
                    g.error_exit("Img size of observer 1 and 2 are different!")
                dsc = dice(
                    test=img_data["1"],
                    reference=img_data["2"],
                    nan_for_nonexisting=False,
                )
                msd = avg_surface_distance_symmetric(
                    test=img_data["1"],
                    reference=img_data["2"],
                    none_for_nonexisting=True,
                    voxel_spacing=g.NII_SPACING,
                )
                hd95 = hausdorff_distance_95(
                    test=img_data["1"],
                    reference=img_data["2"],
                    none_for_nonexisting=True,
                    voxel_spacing=g.NII_SPACING,
                )
                apl = APL(
                    reference_structure=img_data["2"],
                    other_structure=img_data["1"],
                )
                apl_pct = apl.get_apl(normalized=True)
                apl_voxel = apl.get_apl(normalized=False)
                sdsc = surface_dice(
                    test=img_data["1"],
                    reference=img_data["2"],
                    tolerance=1.0,
                )
            # both imgs are empty
            elif img_data["1"] is None and img_data["2"] is None:
                dsc = None
                msd = None
                hd95 = None
                apl_pct = None
                apl_voxel = None
                sdsc = None
            # one of imgs is empty
            else:
                dsc = 0.0
                msd = None
                hd95 = None
                apl_pct = 1.0
                apl_voxel = None
                sdsc = 0.0

            metrics_dict[result_type][patient][Metric.DSC] = dsc
            metrics_dict[result_type][patient][Metric.MSD] = msd
            metrics_dict[result_type][patient][Metric.HD95] = hd95
            metrics_dict[result_type][patient][Metric.APL_PCT] = apl_pct
            metrics_dict[result_type][patient][Metric.APL_VOXEL] = apl_voxel
            metrics_dict[result_type][patient][Metric.SDSC] = sdsc

            # record value for avg/median/max/min calculation
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                metrics_dict[result_type][Stats.AVG][metric].append(
                    metrics_dict[result_type][patient][metric]
                )

        # calculate avg and median
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            metrics_dict[result_type][Stats.MEDIAN][metric] = g.calculate_median(
                metrics_dict[result_type][Stats.AVG][metric]
            )
            metrics_dict[result_type][Stats.MIN][metric] = g.calculate_min(
                metrics_dict[result_type][Stats.AVG][metric]
            )
            metrics_dict[result_type][Stats.MAX][metric] = g.calculate_max(
                metrics_dict[result_type][Stats.AVG][metric]
            )
            # calculate average value at the end
            metrics_dict[result_type][Stats.AVG][metric] = g.calculate_avg(
                metrics_dict[result_type][Stats.AVG][metric]
            )

    # save json
    metrics_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        "iov_{}_vs_{}.json".format(observer["1"], observer["2"]),
    )
    if os.path.exists(metrics_path):
        final_dict = g.load_json(metrics_path)
    else:
        final_dict = Dict()
    final_dict[gtv] = metrics_dict
    g.save_json(data=final_dict, path=metrics_path)


def create_median_table():
    obs_study_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_obs.study")
    observer_mapping = {
        "Obs. 1 vs 2": "iov_Jesper_vs_Kenneth.json",
        "Obs. 2 vs 3": "iov_Kenneth_vs_Hanna.json",
        "Obs. 1 vs 3": "iov_Jesper_vs_Hanna.json",
        "Obs. 1 vs label": "iov_Jesper_vs_Label.json",
        "Obs. 2 vs label": "iov_Kenneth_vs_Label.json",
        "Obs. 3 vs label": "iov_Hanna_vs_Label.json",
    }
    table_data = [["Observers", "DSC", "MSD [mm]", "HD95 [mm]"]]

    for gtv in ["GTVt", "GTVn"]:
        table_data.append([gtv, "", "", ""])

        for obs_pair in observer_mapping:
            new_row = [obs_pair]
            # open json
            json_name = observer_mapping[obs_pair]
            json_data = g.load_json(os.path.join(obs_study_dir, json_name))
            # iov of final correction
            json_data = json_data[gtv.lower()]["correct"]
            # each metric type
            for metric_type in [Metric.DSC, Metric.MSD, Metric.HD95]:
                median_value = json_data[Stats.MEDIAN][metric_type]
                min_value = json_data[Stats.MIN][metric_type]
                max_value = json_data[Stats.MAX][metric_type]
                # rounding off
                decimal_places = 2 if metric_type == Metric.DSC else 1
                # median_value = round(median_value, decimal_places)
                median_value = f"{median_value:.{decimal_places}f}"
                min_value = f"{min_value:.{decimal_places}f}"
                max_value = f"{max_value:.{decimal_places}f}"
                # add metric of current metric type
                new_row.append(f"{median_value} ({min_value}~{max_value})")
            # add new row
            table_data.append(new_row)
        # add empty row after gtvt/gtvn
        table_data.append(["", "", "", ""])

    table_path = os.path.join(obs_study_dir, "iov_median.csv")
    with open(table_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(table_data)


def plot_heatmap():
    observer_list = List(["Jesper", "Kenneth", "Hanna", "Label"])
    label_symbol = ["Obs 1", "Obs 2", "Obs 3", "Label"]
    # label_text = ["Observer 1", "Observer 2", "Observer 3", "Clinical label"]
    # legend_text = "\n".join(
    #     [f"{label_symbol[i]} = {label_text[i]}" for i in range(len(label_text))]
    # )

    obs_study_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_obs.study")
    for gtv in ["gtvt", "gtvn"]:
        if gtv == "gtvt":
            title_gtv = "GTVt"
        elif gtv == "gtvn":
            title_gtv = "GTVn"

        for result_type in [
            # "idl",
            "correct",
        ]:
            # if result_type == "idl":
            #     title_img_name = """"Initial" Segmentations"""
            # elif result_type == "correct":
            #     title_img_name = """"Corrected" Segmentations"""

            # Setting up the figure and axes for a 2x3 grid
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(
                f"Pairwise IOV between observers and clinical label - {title_gtv}"
            )
            # Flattening the axes array for easier iteration
            axes = axes.flatten()

            i = 0
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
            ]:
                if metric in [Metric.SDSC, Metric.DSC]:
                    iov_matrix = np.ones(shape=(4, 4))
                else:
                    iov_matrix = np.zeros(shape=(4, 4))

                for observer_1, observer_2 in observer_list.get_combinations():
                    json_path = os.path.join(
                        obs_study_dir,
                        "iov_{}_vs_{}.json".format(observer_1, observer_2),
                    )
                    data = g.load_json(json_path)
                    data = data[gtv][result_type]["median"][metric]

                    x = observer_list.index(observer_1)
                    y = observer_list.index(observer_2)
                    iov_matrix[x][y] = None
                    iov_matrix[y][x] = data

                # Creating the heatmap with a white-to-blue color gradient
                if metric == Metric.DSC:
                    vmin = 0.7
                    vmax = 1.0
                    cmap = "Blues_r"
                else:
                    vmin = vmax = None
                    cmap = "Blues"

                sns.heatmap(
                    iov_matrix,
                    ax=axes[i],
                    annot=True,
                    cmap=cmap,
                    square=True,
                    cbar=True,
                    vmin=vmin,
                    vmax=vmax,
                    annot_kws={"size": FONT_SIZE},
                )

                subtitle = explain_metric(metric)
                axes[i].set_title(subtitle)
                axes[i].set_xticklabels(
                    label_symbol,
                )
                axes[i].set_yticklabels(
                    label_symbol,
                    rotation=0,
                )

                i += 1

            # turn off axis of the last figure
            # axes[-1].axis("off")
            # fig.delaxes(axes[-1])

            # add legend
            # plt.figtext(
            #     0.96,
            #     0.05,
            #     legend_text,
            #     ha="right",
            #     va="bottom",
            #     fontsize=FONT_SIZE,
            #     bbox={
            #         "facecolor": "white",
            #         "alpha": 1.0,
            #         "pad": 5,
            #         "edgecolor": "gray",
            #     },
            # )

            plt.tight_layout()
            plt.subplots_adjust(top=0.87, wspace=0.25)

            for file_ext in ["pdf", "png"]:
                fig_path = os.path.join(
                    obs_study_dir, f"iov_{gtv}_{result_type}.{file_ext}"
                )
                plt.savefig(fig_path, format=file_ext)


def __calculate_mda_label_iov(patient_observer_map: Dict, gtv: str) -> Dict:
    label_iov = Dict()

    for patient in tqdm(patient_observer_map):

        # add patient_observer mapping of current patient into the list
        cur_patient_labels = Dict()
        for obs in patient_observer_map[patient]:
            cur_observer_label = g.load_gtv_labels(
                dataset_ver=DatasetVer.MDA, patient=f"{patient}_{obs}"
            )[gtv]
            cur_patient_labels[obs] = cur_observer_label

        for obs_1, obs_2 in cur_patient_labels.keys().get_combinations(2):
            label_1 = cur_patient_labels[obs_1]
            label_2 = cur_patient_labels[obs_2]

            obs_pair = f"{obs_1}.vs.{obs_2}"

            label_iov[patient][obs_pair][Metric.DSC] = dice(
                test=label_1,
                reference=label_2,
                nan_for_nonexisting=False,
            )

            label_iov[patient][obs_pair][Metric.MSD] = avg_surface_distance_symmetric(
                test=label_1,
                reference=label_2,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

            label_iov[patient][obs_pair][Metric.HD95] = hausdorff_distance_95(
                test=label_1,
                reference=label_2,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

    # save label iov into json
    g.save_json(
        data=label_iov,
        path=os.path.join(g.DATASET_DIR[DatasetVer.MDA], f"label_iov_{gtv}.json"),
    )

    return label_iov


def __calculate_mda_idl_iov(patient_observer_map: Dict, idl_dir: str, gtv: str) -> Dict:
    idl_iov = Dict()

    for patient in tqdm(patient_observer_map):

        # add patient_observer mapping of current patient into the list
        cur_patient_preds = Dict()
        for obs in patient_observer_map[patient]:
            pred_path = os.path.join(
                idl_dir,
                "patients",
                f"patient={patient}_{obs}",
                "round=01",
                f"{gtv}_pred.nii.gz",
            )
            # load non-existing label
            if not os.path.exists(pred_path):
                g.error_exit(f"{pred_path} doesn't exist")
            else:
                cur_observer_pred = g.load_nii(pred_path, binary=True)
                cur_patient_preds[obs] = cur_observer_pred

        for obs_1, obs_2 in cur_patient_preds.keys().get_combinations(2):
            pred_1 = cur_patient_preds[obs_1]
            pred_2 = cur_patient_preds[obs_2]

            obs_pair = f"{obs_1}.vs.{obs_2}"

            idl_iov[patient][obs_pair][Metric.DSC] = dice(
                test=pred_1,
                reference=pred_2,
                nan_for_nonexisting=False,
            )

            idl_iov[patient][obs_pair][Metric.MSD] = avg_surface_distance_symmetric(
                test=pred_1,
                reference=pred_2,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

            idl_iov[patient][obs_pair][Metric.HD95] = hausdorff_distance_95(
                test=pred_1,
                reference=pred_2,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

    # save label iov into json
    g.save_json(data=idl_iov, path=os.path.join(idl_dir, "idl_iov.json"))

    return idl_iov


# x-axis: iov of labels
# y-axis: iov of idl preds
def plot_mda_label_vs_idl_iov(idl_dir: str):
    if Path(idl_dir).name.startswith("idl.gtvt_"):
        gtv = "gtvt"
    elif Path(idl_dir).name.startswith("idl.gtvn_"):
        gtv = "gtvn"
    else:
        g.error_exit("Invalid idl_dir")

    patient_observer_map = Dict()
    mda_test_set = List(
        g.load_json(g.DATASET_SPLIT_PATH[DatasetVer.MDA])[DatasetPart.TEST]
    )
    # Loop through each element in mda test set:
    for test_set_item in mda_test_set:
        # Split the string into patient ID and observer name
        patient, observer = test_set_item.split("_")
        # create patient key name
        if patient not in patient_observer_map:
            patient_observer_map[patient] = []
        # Add to the dictionary
        patient_observer_map[patient].append(observer)

    # load label iov
    label_iov_json_path = os.path.join(
        g.DATASET_DIR[DatasetVer.MDA], f"label_iov_{gtv}.json"
    )
    # if label iov json file doesn't exist, calculate and create it
    if not os.path.exists(label_iov_json_path):
        label_iov = __calculate_mda_label_iov(
            patient_observer_map=patient_observer_map, gtv=gtv
        )
    else:
        label_iov = g.load_json(label_iov_json_path)

    # load idl iov
    idl_iov_json_path = os.path.join(idl_dir, "idl_iov.json")
    # if idl iov json file doesn't exist, calculate and create it
    if not os.path.exists(idl_iov_json_path):
        idl_iov = __calculate_mda_idl_iov(
            patient_observer_map=patient_observer_map, idl_dir=idl_dir, gtv=gtv
        )
    else:
        idl_iov = g.load_json(idl_iov_json_path)

    # Create plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
    metric_list = [Metric.DSC, Metric.MSD, Metric.HD95]

    for metric_type in tqdm(metric_list):
        x_data = []
        y_data = []

        patient_list = label_iov.keys()
        if patient_list != idl_iov.keys():
            g.error_exit("Mismatch keys between label_iov and idl_iov!")

        for patient in patient_list:

            # Convert keys() of the sub-dict (which is a dict, not a Dict) into List
            # as the sub-dict does not inherit from the custom Dict class.
            obs_pair_list = label_iov[patient].keys()
            if obs_pair_list != idl_iov[patient].keys():
                g.error_exit("Mismatch keys between label_iov and idl_iov!")

            for obs_pair in obs_pair_list:
                x = label_iov[patient][obs_pair][metric_type]
                y = idl_iov[patient][obs_pair][metric_type]
                if g.is_number(x) and g.is_number(y):
                    x_data.append(x)
                    y_data.append(y)

        idx = metric_list.index(metric_type)
        axs[idx].scatter(x_data, y_data)  # , label=explain_metric(metric_type))

        axs[idx].set_title(explain_metric(metric_type))
        # x_label = f"Paired {metric_type.upper()} Between Observers" + (
        #     "" if metric_type == Metric.DSC else " [mm]"
        # )
        axs[idx].set_xlabel("Paired IOV – clinical labels")
        # y_label = f"Post-iDL Paired {metric_type.upper()} Between Observers" + (
        #     "" if metric_type == Metric.DSC else " [mm]"
        # )
        axs[idx].set_ylabel("Paired IOV – post iDL")

        # Joint range calculation for both axes
        all_data = x_data + y_data
        # min_val = min(all_data)
        max_val = max(all_data)
        # margin = (max_val - min_val) * 0.05

        # Set the same range for x and y axes
        if metric_type == Metric.DSC:
            axis_max = 1
            # Ticks as 0.x (2 decimal places)
            ticks = np.round(np.linspace(0, axis_max, num=6), 1)

        elif metric_type == Metric.MSD:
            axis_max = math.floor(max_val)
            while axis_max % 2.5 != 0:
                axis_max += 0.5
            ticks = np.round(np.linspace(0, axis_max, num=6), 1)

        elif metric_type == Metric.HD95:
            axis_max = math.floor(max_val)
            while axis_max % 5 != 0:
                axis_max += 1
            ticks = np.linspace(0, axis_max, num=6, dtype=int)

        axs[idx].set_xlim(0, axis_max)
        axs[idx].set_ylim(0, axis_max)

        # Apply tick settings
        axs[idx].set_xticks(ticks)
        axs[idx].set_yticks(ticks)

        # Add diagonal dashed line
        axs[idx].plot(
            [0, axis_max],
            [0, axis_max],
            linestyle="--",
            color="lime",
            alpha=0.7,
            label="Diagonal",
        )

    fig.suptitle(
        f"Effect of iDL on IOV – {gtv[:3].upper() + gtv[3]} (MDA dataset)"
    )
    plt.tight_layout()

    # Save the figure in multiple formats
    for file_ext in ["pdf", "png"]:
        fig_path = os.path.join(idl_dir, f"iov_label_vs_idl.{file_ext}")
        plt.savefig(fig_path, format=file_ext)

    plt.show()
