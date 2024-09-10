import csv
import os

import cv2
import global_utils.global_core as g
import matplotlib.pyplot as plt
import numpy as np
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import Metric, ObsStudyStep, Plane, Stat
from metric_utils.added_path_len import APL
from metric_utils.metric_func import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
    surface_distances,
)
from research_utils.research_core import COLOR_LIST, explain_metric
from tqdm import tqdm


def calculate_comparison_metrics(obs_study_id: str):
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
        for stat in [Stat.AVG, Stat.MEDIAN]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
            ]:
                plane_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
                if metric == Metric.MSD or metric == Metric.HD95:
                    plane_list.append("anatomical")
                for plane in plane_list:
                    metrics_dict[stat][metric][plane] = []
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
            if obs_study_step[patient] == ObsStudyStep.APPROVED:
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

            # for anatomical msd and hd95
            sds_full = None
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

                # calculate anatomical msd and hd95
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

                # record value for avg and median calculation
                for stat in [Stat.AVG, Stat.MEDIAN]:
                    for metric in [
                        Metric.DSC,
                        Metric.MSD,
                        Metric.HD95,
                        Metric.APL_PCT,
                        Metric.APL_VOXEL,
                    ]:
                        metrics_dict[stat][metric][plane].append(
                            metrics_dict[patient][metric][plane]
                        )

            # anatomical msd and hd95
            metrics_dict[patient][Metric.MSD]["anatomical"] = np.mean(sds_full)
            metrics_dict[patient][Metric.HD95]["anatomical"] = np.percentile(
                sds_full, 95
            )
            # record value for avg and median calculation
            for stat in [Stat.AVG, Stat.MEDIAN]:
                for metric in [Metric.MSD, Metric.HD95]:
                    metrics_dict[stat][metric]["anatomical"].append(
                        metrics_dict[patient][metric]["anatomical"]
                    )

        # calculate avg and median
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
        ]:
            plane_list = [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]
            if metric == Metric.MSD or metric == Metric.HD95:
                plane_list.append("anatomical")
            for plane in plane_list:
                metrics_dict[Stat.MEDIAN][metric][plane] = g.calculate_median(
                    metrics_dict[Stat.MEDIAN][metric][plane]
                )
                metrics_dict[Stat.AVG][metric][plane] = g.calculate_avg(
                    metrics_dict[Stat.AVG][metric][plane]
                )

        g.save_json(data=metrics_dict, path=metrics_path)


def create_metrics_comparison_table(obs_study_id_list: list):
    target_pairs = [
        ("idl", "correct"),
        ("delineation", "idl"),
        ("delineation", "correct"),
    ]

    for target_1, target_2 in target_pairs:

        table_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "2d_{}_vs_{}.csv".format(target_1, target_2),
        )
        # Header row
        table_data = [
            [
                "Metric",
                "Anatomical Plane",
                "Statistics",
                "Jesper",
                "Kenneth",
                "Hanna",
            ],
        ]
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            # Metric.SDSC,
        ]:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                for stat in [
                    # Stat.AVG,
                    Stat.MEDIAN,
                ]:
                    cur_item = [metric, plane, stat, "", "", ""]
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

            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                # Metric.SDSC,
            ]:
                for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                    for stat in [
                        # Stat.AVG,
                        Stat.MEDIAN,
                    ]:
                        cur_value = metrics_dict[stat][metric][plane]
                        for item in table_data:
                            if (
                                item[0] == metric
                                and item[1] == plane
                                and item[2] == stat
                            ):
                                if "Jesper" in obs_study_id:
                                    item[3] = cur_value
                                elif "Kenneth" in obs_study_id:
                                    item[4] = cur_value
                                elif "Hanna" in obs_study_id:
                                    item[5] = cur_value
                                break

        with open(table_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(table_data)


def calculate_input_inconsistency(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        g.error_exit("Must be an 'idl.gtvt' id!")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    metrics_dict = Dict()
    metrics_path = os.path.join(obs_study_dir, "gtvt_input_inconsistency.json")
    for stat in [Stat.AVG, Stat.MEDIAN]:
        metrics_dict[stat] = []
        # for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
        #     metrics_dict[stat][plane] = []
    g.save_json(data=metrics_dict, path=metrics_path)

    # get patients
    patient_list = []
    # open "obs_study_step.json" and find approved patients
    obs_study_step = g.load_json(
        os.path.join(
            obs_study_dir,
            "obs_study_step.json",
        )
    )
    for patient in obs_study_step.keys():
        if obs_study_step[patient] == ObsStudyStep.APPROVED:
            patient_list.append(patient)

    # loop through patients
    for patient in tqdm(patient_list):
        patient_dir = os.path.join(obs_study_dir, "patients", patient)
        idl_img_dir = os.path.join(patient_dir, "round=01")

        delineation_3d = g.load_nii(
            os.path.join(idl_img_dir, "gtvt_delineation.nii.gz"),
            binary=True,
        )
        # delineation_2d = Dict()
        selected_slices = g.load_json(os.path.join(patient_dir, "selected_slices.json"))

        # hd100_max = None
        sds_full = None

        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            slice_id = int(selected_slices[plane]["round=01"])

            if plane == Plane.TRANSVERSE:
                delineation_2d = delineation_3d[slice_id, :, :]
            elif plane == Plane.CORONAL:
                delineation_2d = delineation_3d[:, slice_id, :]
            elif plane == Plane.SAGITTAL:
                delineation_2d = delineation_3d[:, :, slice_id]

            kernel = np.ones((3, 3), np.uint8)
            delineation_2d_fixed = cv2.morphologyEx(
                delineation_2d, cv2.MORPH_OPEN, kernel
            )
            delineation_2d_fixed = cv2.morphologyEx(
                delineation_2d_fixed, cv2.MORPH_CLOSE, kernel
            )

            # hd100 = hausdorff_distance(
            #     test=delineation_2d,
            #     reference=delineation_2d_fixed,
            #     none_for_nonexisting=True,
            #     voxel_spacing=(1, 1),  # g.NII_SPACING,
            # )
            # if hd100_max is None:
            #     hd100_max = hd100
            # elif hd100 > hd100_max:
            #     hd100_max = hd100

            sds1 = surface_distances(
                binary_img_1=delineation_2d,
                binary_img_2=delineation_2d_fixed,
                spacing=(1, 1),
            )
            sds2 = surface_distances(
                binary_img_1=delineation_2d_fixed,
                binary_img_2=delineation_2d,
                spacing=(1, 1),
            )
            sds = np.hstack((sds1, sds2))
            if sds_full is None:
                sds_full = sds
            else:
                sds_full = np.hstack((sds_full, sds))

        # hd100 of 3 planes
        hd100 = max(sds_full)
        # hd95 = np.percentile(sds_full, 95)
        metrics_dict[patient] = round(hd100)

        # record value for avg and median calculation
        for stat in [Stat.AVG, Stat.MEDIAN]:
            metrics_dict[stat].append(hd100)

    # calculate avg and median
    metrics_dict[Stat.MEDIAN] = g.calculate_median(metrics_dict[Stat.MEDIAN])
    metrics_dict[Stat.AVG] = g.calculate_avg(metrics_dict[Stat.AVG])

    g.save_json(data={"hd100": metrics_dict}, path=metrics_path)


def plot_metrics_comparison(obs_study_id_list: list):
    target_pairs = [
        # ("idl", "correct"),
        ("delineation", "idl"),
        # ("delineation", "correct"),
    ]

    for target_1, target_2 in target_pairs:

        fig_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "2d_{}_vs_{}.pdf".format(target_1, target_2),
        )

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        # Configure title and labels
        if target_1 == "delineation":
            str_1 = "User Input"
        elif target_1 == "idl":
            str_1 = """"Initial" Segmentation"""
        elif target_1 == "correct":
            str_1 = """"Corrected" Segmentation"""

        if target_2 == "delineation":
            str_2 = "User Input"
        elif target_2 == "idl":
            str_2 = """"Initial" Segmentation"""
        elif target_2 == "correct":
            str_2 = """"Corrected" Segmentation"""

        fig.suptitle(
            "{} vs {} (on selected GTVt slices)".format(str_1, str_2),
        )
        axes = axes.flatten()

        i = 0
        for metric in [Metric.MSD, Metric.HD95]:

            # Create a grouped bar plot
            ax = axes[i]

            for obs_study_id in tqdm(obs_study_id_list):

                x_list = List()
                y_list = List()

                if not obs_study_id.startswith("idl.gtvt_"):
                    g.error_exit("Must be an 'idl.gtvt' id!")

                if "Jesper" in obs_study_id:
                    observer = "Observer 1"
                    color = COLOR_LIST[0]
                elif "Kenneth" in obs_study_id:
                    observer = "Observer 2"
                    color = COLOR_LIST[1]
                elif "Hanna" in obs_study_id:
                    observer = "Observer 3"
                    color = COLOR_LIST[2]

                obs_study_dir = os.path.join(
                    g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
                )
                x_axis_dict = g.load_json(
                    os.path.join(obs_study_dir, "gtvt_input_inconsistency.json")
                )["hd100"]
                y_axis_dict = g.load_json(
                    os.path.join(
                        obs_study_dir, "2d_{}_vs_{}.json".format(target_1, target_2)
                    )
                )

                # get patients
                patient_list = []
                # open "obs_study_step.json" and find approved patients
                obs_study_step = g.load_json(
                    os.path.join(
                        obs_study_dir,
                        "obs_study_step.json",
                    )
                )
                for patient in obs_study_step.keys():
                    if obs_study_step[patient] == ObsStudyStep.APPROVED:
                        patient_list.append(patient)

                # loop through patients
                for patient in patient_list:
                    x_value = x_axis_dict[patient]
                    x_list.append(x_value)

                    y_value = y_axis_dict[patient][metric]["anatomical"]
                    y_list.append(y_value)

                # Perform linear regression
                # Sort x_list and get the indices of the sorted order
                sorted_indices = np.argsort(x_list)
                x_list = np.array(x_list)[sorted_indices]
                # Reorder y_list using the sorted indices
                y_list = [y_list[i] for i in sorted_indices]
                x_list = np.array(x_list)
                y_list = np.array(y_list)
                m, b = np.polyfit(x_list, y_list, 1)

                # Scatter plot
                ax.scatter(
                    x=x_list,
                    y=y_list,
                    label=observer,
                    color=color,
                )
                # Plot the regression line
                ax.plot(
                    x_list,
                    m * x_list + b,
                    color=color,
                )

            ax.set_title(explain_metric(metric))

            # ax.set_ylabel(
            #     ylabel=metric.upper(),
            #     rotation=0,
            #     position=(0, 1),
            #     va="bottom",
            #     labelpad=-9,
            # )
            ax.set_xlabel(
                xlabel="GTVt Input Inconsistency",
            )

            # Add a legend to describe the observers
            ax.legend(loc="upper left")

            # # Ensuring the plot is square
            # ax.set_aspect("equal")

            # next sub fig
            i += 1

        # # Adjust layout to prevent overlap and save the entire figure as a PDF
        plt.tight_layout()
        # Save the plot as a PDF file in the specified directory
        plt.savefig(fig_path, format="pdf")
