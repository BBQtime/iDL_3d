import csv
import os

import cv2
import global_utils.global_core as g
import matplotlib

# Prevent matplotlib.pyplot from using a GUI (like X11) for rendering.
# Without this line, using breakpoints under X11 without VCXSRV can cause the debugger to freeze.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import (
    DatasetPart,
    DatasetVer,
    Metric,
    ObsStudyGTVtStep,
    Plane,
    Stats,
)
from metric_utils.added_path_len import APL
from metric_utils.metric_func import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
    surface_dice,
    surface_distances,
)
from research_utils.research_core import (
    COLOR_LIST,
    explain_metric,
    get_obs_study_patients_list,
)
from tqdm import tqdm


def calculate_metrics(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        g.error_exit("Must be an 'idl.gtvt' id!")

    target_pairs = [
        ("idl", "correct"),
        ("delineation", "idl"),
        ("delineation", "correct"),
    ]

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )
    for target_1, target_2 in target_pairs:
        metrics_dict = Dict()
        metrics_path = os.path.join(
            obs_study_dir, "2d_{}_vs_{}.json".format(target_1, target_2)
        )
        for stats in [Stats.AVG, Stats.MEDIAN]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                plane_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                if metric == Metric.MSD or metric == Metric.HD95:
                    plane_list.append("cross.plane")
                for plane in plane_list:
                    metrics_dict[stats][metric][plane] = []
        g.save_json(data=metrics_dict, path=metrics_path)

        patient_list = []
        # open "obs_study_step.json" and find approved patients
        obs_study_step = g.load_json(
            os.path.join(
                obs_study_dir,
                "obs_study_step.json",
            )
        )
        for patient in obs_study_step.keys():
            # patient 462 is for testing
            if patient == "patient=462":
                continue
            if obs_study_step[patient]["gtvt"] == ObsStudyGTVtStep.APPROVED:
                patient_list.append(patient)

        # loop through patients
        for patient in tqdm(patient_list):
            patient_dir = os.path.join(obs_study_dir, "patients", patient)
            idl_img_dir = os.path.join(patient_dir, "round=01")

            delineation = g.load_nii(
                os.path.join(idl_img_dir, "gtvt_delineation.nii.gz"),
                binary=True,
            )
            origin_pred = g.load_nii(
                os.path.join(idl_img_dir, "gtvt_pred.nii.gz"),
                binary=True,
            )
            correction_mask = g.load_nii(
                os.path.join(idl_img_dir, "gtvt_correction_mask.nii.gz"),
                binary=True,
            )
            correction = g.load_nii(
                os.path.join(idl_img_dir, "gtvt_correction.nii.gz"),
                binary=True,
            )
            final_pred = g.combine_pred_correction(
                origin_pred=origin_pred,
                correction=correction,
                correction_mask=correction_mask,
            )

            selected_slices = g.load_json(
                os.path.join(patient_dir, "selected_slices.json")
            )

            sds_full = None  # for cross-plane msd and hd95
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                slice_id = int(selected_slices[plane]["round=01"])

                if plane == Plane.TRANSVERSE:
                    delineation_2d = delineation[slice_id, :, :]
                    origin_pred_2d = origin_pred[slice_id, :, :]
                    final_pred_2d = final_pred[slice_id, :, :]
                elif plane == Plane.CORONAL:
                    delineation_2d = delineation[:, slice_id, :]
                    origin_pred_2d = origin_pred[:, slice_id, :]
                    final_pred_2d = final_pred[:, slice_id, :]
                elif plane == Plane.SAGITTAL:
                    delineation_2d = delineation[:, :, slice_id]
                    origin_pred_2d = origin_pred[:, :, slice_id]
                    final_pred_2d = final_pred[:, :, slice_id]

                kernel = np.ones((3, 3), np.uint8)
                delineation_2d = cv2.morphologyEx(
                    delineation_2d, cv2.MORPH_OPEN, kernel
                )
                delineation_2d = cv2.morphologyEx(
                    delineation_2d, cv2.MORPH_CLOSE, kernel
                )

                # test
                if target_1 == "delineation":
                    test = delineation_2d
                elif target_1 == "idl":
                    test = origin_pred_2d
                elif target_1 == "correct":
                    test = final_pred_2d

                # reference
                if target_2 == "delineation":
                    reference = delineation_2d
                elif target_2 == "idl":
                    reference = origin_pred_2d
                elif target_2 == "correct":
                    reference = final_pred_2d

                # dsc/msd/hd95
                dsc = dice(
                    test=test,
                    reference=reference,
                    nan_for_nonexisting=False,
                )
                msd = avg_surface_distance_symmetric(
                    test=test,
                    reference=reference,
                    none_for_nonexisting=True,
                    voxel_spacing=(1, 1),  # g.NII_SPACING,
                )
                hd95 = hausdorff_distance_95(
                    test=test,
                    reference=reference,
                    none_for_nonexisting=True,
                    voxel_spacing=(1, 1),  # g.NII_SPACING,
                )

                metrics_dict[patient][Metric.DSC][plane] = dsc
                metrics_dict[patient][Metric.MSD][plane] = msd
                metrics_dict[patient][Metric.HD95][plane] = hd95

                # calculate cross-plane msd and hd95
                sds1 = surface_distances(
                    binary_img_1=test,
                    binary_img_2=reference,
                    spacing=(1, 1),
                )
                sds2 = surface_distances(
                    binary_img_1=reference,
                    binary_img_2=test,
                    spacing=(1, 1),
                )
                sds = np.hstack((sds1, sds2))
                if sds_full is None:
                    sds_full = sds
                else:
                    sds_full = np.hstack((sds_full, sds))

                # added path length
                apl = APL(reference_structure=reference, other_structure=test)
                apl_pct = apl.get_apl(normalized=True)
                apl_voxel = apl.get_apl(normalized=False)
                metrics_dict[patient][Metric.APL_PCT][plane] = apl_pct
                metrics_dict[patient][Metric.APL_VOXEL][plane] = apl_voxel

                # surface dice
                sdsc = surface_dice(
                    test=test,
                    reference=reference,
                    tolerance=1.0,
                )
                # tensor to float
                metrics_dict[patient][Metric.SDSC][plane] = sdsc

                # record value for avg and median calculation
                for stats in [Stats.AVG, Stats.MEDIAN]:
                    for metric in [
                        Metric.DSC,
                        Metric.MSD,
                        Metric.HD95,
                        Metric.APL_PCT,
                        Metric.APL_VOXEL,
                        Metric.SDSC,
                    ]:
                        metrics_dict[stats][metric][plane].append(
                            metrics_dict[patient][metric][plane]
                        )

            # calculate cross-plane msd and hd95
            metrics_dict[patient][Metric.MSD]["cross.plane"] = np.mean(sds_full)
            metrics_dict[patient][Metric.HD95]["cross.plane"] = np.percentile(
                sds_full, 95
            )
            # record value for avg and median calculation
            for stats in [Stats.AVG, Stats.MEDIAN]:
                for metric in [Metric.MSD, Metric.HD95]:
                    metrics_dict[stats][metric]["cross.plane"].append(
                        metrics_dict[patient][metric]["cross.plane"]
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
            plane_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
            if metric == Metric.MSD or metric == Metric.HD95:
                plane_list.append("cross.plane")
            for plane in plane_list:
                metrics_dict[Stats.MEDIAN][metric][plane] = g.calculate_median(
                    metrics_dict[Stats.MEDIAN][metric][plane]
                )
                metrics_dict[Stats.AVG][metric][plane] = g.calculate_avg(
                    metrics_dict[Stats.AVG][metric][plane]
                )

        g.save_json(data=metrics_dict, path=metrics_path)


def create_metrics_tables(obs_study_id_list: list):
    target_pairs = [
        ("idl", "correct"),
        ("delineation", "idl"),
        ("delineation", "correct"),
    ]

    patients_list = get_obs_study_patients_list()
    patients_list = List([Stats.AVG, Stats.MEDIAN]) + patients_list

    for target_1, target_2 in target_pairs:
        print(f"2D {target_1} vs {target_2}")
        table_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "2d_{}_vs_{}.csv".format(target_1, target_2),
        )
        # Header row
        table_data = [
            [
                "Patient",
                "Metric (Cross-Plane)",
                "Jesper",
                "Kenneth",
                "Hanna",
            ],
        ]

        for patient in patients_list:
            for metric_type in [
                # Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                # Metric.APL_PCT,
                # Metric.APL_VOXEL,
                # Metric.SDSC,
            ]:
                cur_item = [patient, metric_type, "", "", ""]
                table_data.append(cur_item)

        for obs_study_id in tqdm(obs_study_id_list):
            if not obs_study_id.startswith("idl.gtvt_"):
                g.error_exit("Must be an 'idl.gtvt' id!")

            obs_study_dir = os.path.join(
                g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
            )
            metrics_dict = g.load_json(
                os.path.join(
                    obs_study_dir, "2d_{}_vs_{}.json".format(target_1, target_2)
                )
            )
            for patient in patients_list:
                for metric_type in [
                    # Metric.DSC,
                    Metric.MSD,
                    Metric.HD95,
                    # Metric.APL_PCT,
                    # Metric.APL_VOXEL,
                    # Metric.SDSC,
                ]:
                    if patient not in [Stats.AVG, Stats.MEDIAN]:
                        cur_value = metrics_dict[f"patient={patient}"]
                    else:
                        cur_value = metrics_dict[patient]
                    cur_value = cur_value[metric_type]["cross.plane"]

                    for item in table_data:
                        if item[0] == patient and item[1] == metric_type:
                            if "Jesper" in obs_study_id:
                                item[2] = cur_value
                            elif "Kenneth" in obs_study_id:
                                item[3] = cur_value
                            elif "Hanna" in obs_study_id:
                                item[4] = cur_value
                            break

        with open(table_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(table_data)


def plot_bias_gtvt_center():
    metrics = [Metric.DSC, Metric.MSD, Metric.HD95]

    data = Dict()
    baseline_au_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_au")
    bias_results_dirs = g.get_sub_dirs(
        baseline_au_dir,
        key_word="au.ext_gravity.center.bias.range",
        full_path=True,
    )

    x_axis_labels = []

    # Loop through bias results dirs and load metrics of each patient
    for bias_result_dir in bias_results_dirs:
        # if "gravity.center.bias.range=0" in bias_result_dir:
        #     continue

        bias_range = g.load_json(os.path.join(bias_result_dir, "hyper.json"))[
            "gravity.center.bias.range"
        ]
        x_axis_labels.append(bias_range)

        metric_json_path = os.path.join(bias_result_dir, "inference_au.ext_test.json")
        metric_dict = g.load_json(metric_json_path)
        for metric_type in [Metric.DSC, Metric.MSD, Metric.HD95]:
            if bias_range not in data:
                data[bias_range] = {}
            if metric_type not in data[bias_range]:
                data[bias_range][metric_type] = []
            for patient in metric_dict.keys():
                if patient in [Stats.AVG, Stats.MEDIAN]:
                    continue
                cur_value = metric_dict[patient][metric_type]["round=01"]
                if cur_value is not None:  # Ensure the value is not None
                    data[bias_range][metric_type].append(cur_value)

    # # CoG baseline values
    # cog_baseline = {
    #     Metric.DSC: 0.867687026176512,
    #     Metric.MSD: 1.2264917672792,
    #     Metric.HD95: 3.60555127546398,
    # }

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle("iDL with Biased CoG(Center of Gravity) for GTVt", fontsize=16)
    colors = plt.cm.tab10(np.linspace(0, 1, len(x_axis_labels)))

    for i, metric_type in enumerate(metrics):
        ax = axes[i]

        # Collect data for the current metric
        metric_data = [data[bias_range][metric_type] for bias_range in x_axis_labels]

        # Boxplot for each experiment with individual colors
        for idx, experiment_data in enumerate(metric_data):
            ax.boxplot(
                [experiment_data],
                positions=[idx],
                patch_artist=True,
                boxprops=dict(facecolor=colors[idx % len(colors)]),
                widths=0.5,
                medianprops=dict(color="white", linewidth=2),
            )

        # # Add baseline as dashed line
        # ax.axhline(
        #     y=cog_baseline[metric_type],
        #     color="black",
        #     linestyle="--",
        #     linewidth=2,
        #     label="CoG Baseline",
        # )

        # Formatting
        ax.set_title(explain_metric(metric_type))
        ax.set_xlabel("Bias Range (within ±voxels)")
        ax.set_ylabel("Metric Value")
        ax.set_xticks(ticks=range(len(x_axis_labels)), labels=x_axis_labels)

        if i == 0:  # Add legend to the first subplot
            ax.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    for file_ext in ["pdf", "png"]:
        fig_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            f"bias_gtvt_center.{file_ext}",
        )
        plt.savefig(fig_path, format=file_ext)
