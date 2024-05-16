import csv
import os

import cv2
import global_core as g
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from added_path_len import APL
from custom_dict import Dict
from custom_list import List
from segment_metric import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance,
    hausdorff_distance_95,
    surface_dice,
    surface_distances,
)
from str_lib import Metric, ObsStudyStep, Plane, Stat
from tqdm import tqdm


def explain_metric(metric: str):
    if metric == Metric.DSC:
        return "DSC"
    elif metric == Metric.MSD:
        return "Mean Surface Distance"
    elif metric == Metric.HD95:
        return "Hausdorff Distance 95"
    elif metric == Metric.APL_PCT:
        return "Added Path Length (Normalized)"
    elif metric == Metric.APL_VOXEL:
        return "Added Path Length (Voxels)"
    elif metric == Metric.SDSC:
        return "Surface DSC (1mm)"


def calculate_3d_idl_vs_correct(obs_study_id: str):

    if obs_study_id.startswith("idl.gtvn_"):
        gtv = "gtvn"
    elif obs_study_id.startswith("idl.gtvt_"):
        gtv = "gtvt"
    else:
        g.error_exit("obs study train id error")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    metrics_dict = Dict()
    metrics_path = os.path.join(obs_study_dir, "3d_idl_vs_correct.json")
    for stat in [Stat.AVG, Stat.MEDIAN]:
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            metrics_dict[stat][metric] = []
    g.save_json(data=metrics_dict, path=metrics_path)

    patient_list = []
    # open "obs_study_step.json" and find approved patients
    obs_study_step_json_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        g.replace_char_in_str(obs_study_id, 7, "t"),
        "obs_study_step.json",
    )
    obs_study_step = g.load_json(obs_study_step_json_path)
    for patient in obs_study_step.keys():
        if obs_study_step[patient] == ObsStudyStep.APPROVED:
            patient_list.append(patient)

    for patient in tqdm(patient_list):

        patient_dir = os.path.join(obs_study_dir, "patients", patient)

        if os.path.exists(patient_dir):
            idl_img_dir = os.path.join(patient_dir, "round=01")
            origin_pred = g.load_nii(
                os.path.join(idl_img_dir, "{}_pred.nii.gz".format(gtv)),
                binary=True,
            )
            correction_mask = g.load_nii(
                os.path.join(idl_img_dir, "{}_correction_mask.nii.gz".format(gtv)),
                binary=True,
            )
            correction = g.load_nii(
                os.path.join(idl_img_dir, "{}_correction.nii.gz".format(gtv)),
                binary=True,
            )
            final_pred = g.combine_pred_correction(
                origin_pred=origin_pred,
                correction=correction,
                correction_mask=correction_mask,
            )
        else:
            final_pred = None
            origin_pred = None

        reference = final_pred
        test = origin_pred

        # dsc/msd/hd95
        if os.path.exists(patient_dir):
            dsc = dice(
                test=test,
                reference=reference,
                nan_for_nonexisting=False,
            )
            msd = avg_surface_distance_symmetric(
                test=test,
                reference=reference,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )
            hd95 = hausdorff_distance_95(
                test=test,
                reference=reference,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )
        else:
            dsc = 1.0
            msd = 0.0
            hd95 = 0.0

        metrics_dict[patient][Metric.DSC] = dsc
        metrics_dict[patient][Metric.MSD] = msd
        metrics_dict[patient][Metric.HD95] = hd95

        # added path length
        if os.path.exists(patient_dir):
            apl = APL(reference_structure=reference, other_structure=test)
            apl_pct = apl.get_apl(normalized=True)
            apl_voxel = apl.get_apl(normalized=False)
        else:
            apl_pct = 0.0
            apl_voxel = 0

        metrics_dict[patient][Metric.APL_PCT] = apl_pct
        metrics_dict[patient][Metric.APL_VOXEL] = apl_voxel

        # surface dice
        if os.path.exists(patient_dir):
            sdsc = surface_dice(
                test=test,
                reference=reference,
                tolerance=1.0,
            )
            # tensor to float
            metrics_dict[patient][Metric.SDSC] = sdsc
        else:
            metrics_dict[patient][Metric.SDSC] = 1.0

        # record value for avg and median calculation
        for stat in [Stat.AVG, Stat.MEDIAN]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                metrics_dict[stat][metric].append(metrics_dict[patient][metric])

    # calculate avg and median
    for metric in [
        Metric.DSC,
        Metric.MSD,
        Metric.HD95,
        Metric.APL_PCT,
        Metric.APL_VOXEL,
        Metric.SDSC,
    ]:
        metrics_dict[Stat.MEDIAN][metric] = g.calculate_median(
            metrics_dict[Stat.MEDIAN][metric]
        )
        metrics_dict[Stat.AVG][metric] = g.calculate_avg(metrics_dict[Stat.AVG][metric])

    g.save_json(data=metrics_dict, path=metrics_path)


def create_table_3d_idl_vs_correct(obs_study_id_list: list):
    table_path = Dict()
    table_data = Dict()

    for i in ["gtvt", "gtvn"]:
        table_path[i] = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "3d_idl_vs_correct_{}.csv".format(i),
        )

        table_data[i] = [
            ["Metric", "Statistics", "Jesper", "Kenneth", "Hanna"],  # Header row
        ]
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            for stat in [
                # Stat.AVG,
                Stat.MEDIAN,
            ]:
                cur_item = [metric, stat, "", "", ""]
                table_data[i].append(cur_item)

    for obs_study_id in tqdm(obs_study_id_list):
        if obs_study_id.startswith("idl.gtvn_"):
            cur_table_data = table_data["gtvn"]
        elif obs_study_id.startswith("idl.gtvt_"):
            cur_table_data = table_data["gtvt"]
        else:
            g.error_exit("obs study train id error")

        obs_study_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
        )
        metrics_dict = g.load_json(
            os.path.join(obs_study_dir, "3d_idl_vs_correct.json")
        )

        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            for stat in [
                # Stat.AVG,
                Stat.MEDIAN,
            ]:
                cur_value = metrics_dict[stat][metric]
                for item in cur_table_data:
                    if item[0] == metric and item[1] == stat:
                        if "Jesper" in obs_study_id:
                            item[2] = cur_value
                        elif "Kenneth" in obs_study_id:
                            item[3] = cur_value
                        elif "Hanna" in obs_study_id:
                            item[4] = cur_value
                        break

    for i in ["gtvt", "gtvn"]:
        with open(table_path[i], "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(table_data[i])


def plot_3d_idl_vs_correct(obs_study_id_list: list):
    gtv = None
    for obs_study_id in obs_study_id_list:
        if obs_study_id.startswith("idl.gtvn_"):
            cur_gtv = "gtvn"
        elif obs_study_id.startswith("idl.gtvt_"):
            cur_gtv = "gtvt"
        else:
            g.error_exit("obs study train id error")

        if gtv is None:
            gtv = cur_gtv
        elif gtv != cur_gtv:
            g.error_exit("All items in obs_study_id_list should be same gtv type")

    fig_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        "3d_idl_vs_correct_{}.pdf".format(gtv),
    )

    patients_list = ["489", "496", "499", "509", "513", "536", "538"]
    observers_list = ["Jesper", "Kenneth", "Hanna"]

    # Set up a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    if gtv == "gtvt":
        title_gtv = "GTVt"
    elif gtv == "gtvn":
        title_gtv = "GTVn"
    fig.suptitle(
        "{} Initial Segmentation vs Final Correction".format(title_gtv), fontsize=20
    )
    axes = axes.flatten()

    i = 0
    for metric in tqdm(
        [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            # Metric.APL_VOXEL,
            Metric.SDSC,
        ]
    ):

        fig_data = Dict()
        for observer in observers_list:
            fig_data[observer] = []

        # loop through observer study train id
        for obs_study_id in obs_study_id_list:

            metrics_dict = g.load_json(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    "baseline_obs.study",
                    obs_study_id,
                    "3d_idl_vs_correct.json",
                )
            )

            # get observer name from train id
            for observer in observers_list:
                if observer in obs_study_id:
                    break

            for patient in patients_list:
                fig_data[observer].append(
                    metrics_dict["patient={}".format(patient)][metric]
                )

        ax = axes[i]

        # Define bar width for clarity in grouped bars
        bar_width = 0.25

        # Calculate indices for x-axis where groups of bars will be located
        indices = np.arange(len(patients_list))

        # Plot bars for each observer
        for observer in observers_list:
            idx = fig_data.key_index(observer)
            ax.bar(
                x=indices + idx * bar_width,  # list
                height=fig_data[observer],  # list
                width=bar_width,
                label="Observer {}".format(observers_list.index(observer) + 1),
            )

        # Configure title and labels
        # ax.set_title("Initial Segmentation vs Final Correction")
        ax.set_title(explain_metric(metric))
        # ax.set_ylabel(
        #     ylabel=explain_metric(metric),
        #     # rotation=0,
        #     # position=(0, 1),
        #     # va="bottom",
        #     # labelpad=-9,
        # )
        ax.set_xlabel(
            xlabel="Patient",
            # position=(1, 0),
            # labelpad=-11,
        )

        # Define y-axis range to accommodate label placement above 1.0
        # Define y-axis ticks to display key points including 1.0
        if metric == Metric.DSC or metric == Metric.SDSC or metric == Metric.APL_PCT:
            ax.set_ylim(0, 1.1)
            # ax.set_yticks(np.linspace(0, 1, 6))

        # Set x-axis ticks to be centered under each group of bars
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])

        # Add a legend to describe the observers
        legend = ax.legend(loc="best")  # "upper right")
        legend.get_frame().set_alpha(0.3)

        # next sub plot
        i += 1

    # turn off axis of the last figure
    # axes[-1].axis("off")
    fig.delaxes(axes[-1])

    # Adjust layout to prevent overlap and save the entire figure as a PDF
    plt.tight_layout()
    # Save the plot as a PDF file in the specified directory
    plt.savefig(fig_path, format="pdf")


def calculate_gtvt_slices_metrics(obs_study_id: str):
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


def create_table_gtvt_slices_metrics(obs_study_id_list: list):
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


def calculate_gtvt_input_variation(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        g.error_exit("Must be an 'idl.gtvt' id!")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    metrics_dict = Dict()
    metrics_path = os.path.join(obs_study_dir, "gtvt_input_variation.json")
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


def plot_gtvt_slices_metrics(obs_study_id_list: list):
    target_pairs = [
        ("idl", "correct"),
        ("delineation", "idl"),
        ("delineation", "correct"),
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
            str_1 = "Initial Segmentation"
        elif target_1 == "correct":
            str_1 = "Final Correction"

        if target_2 == "delineation":
            str_2 = "User Input"
        elif target_2 == "idl":
            str_2 = "Initial Segmentation"
        elif target_2 == "correct":
            str_2 = "Final Correction"

        fig.suptitle(
            "Selected GTVt slices - {} vs {}".format(str_1, str_2), fontsize=20
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
                elif "Kenneth" in obs_study_id:
                    observer = "Observer 2"
                elif "Hanna" in obs_study_id:
                    observer = "Observer 3"

                obs_study_dir = os.path.join(
                    g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
                )
                x_axis_dict = g.load_json(
                    os.path.join(obs_study_dir, "gtvt_input_variation.json")
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
                )
                # Plot the regression line
                ax.plot(x_list, m * x_list + b)

            ax.set_title("Anatomical " + explain_metric(metric))

            # ax.set_ylabel(
            #     ylabel=metric.upper(),
            #     rotation=0,
            #     position=(0, 1),
            #     va="bottom",
            #     labelpad=-9,
            # )
            ax.set_xlabel(
                xlabel="Anatomical Plane Variation (HD 100) of GTVt User Input",
            )

            # Add a legend to describe the observers
            ax.legend()

            # # Ensuring the plot is square
            # ax.set_aspect("equal")

            # next sub fig
            i += 1

        # Save the plot as a PDF file in the specified directory
        plt.savefig(fig_path, format="pdf")


def calculate_iov(obs_study_id_1: str, obs_study_id_2: str):
    obs_study_id = Dict()
    obs_study_id["1"] = obs_study_id_1
    obs_study_id["2"] = obs_study_id_2

    if obs_study_id["1"] == obs_study_id["2"]:
        g.error_exit("2 input obs_study_ids cannot be identical.")

    if obs_study_id["1"].startswith("idl.gtvn_") and obs_study_id["2"].startswith(
        "idl.gtvn_"
    ):
        gtv = "gtvn"
    elif obs_study_id["1"].startswith("idl.gtvt_") and obs_study_id["2"].startswith(
        "idl.gtvt_"
    ):
        gtv = "gtvt"
    else:
        g.error_exit("obs study train id error")

    observer = Dict()
    # get observer from obs study id
    for name in ["Jesper", "Kenneth", "Hanna"]:
        for idx in ["1", "2"]:
            if name in obs_study_id[idx]:
                observer[idx] = name
    if observer["1"] == observer["2"]:
        g.error_exit("2 observers cannot be identical.")

    obs_study_dir = Dict()
    for idx in ["1", "2"]:
        obs_study_dir[idx] = os.path.join(
            g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id[idx]
        )

    metrics_dict = Dict()
    metrics_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        "iov_{}_vs_{}.json".format(observer["1"], observer["2"]),
    )

    for img_name in ["idl", "correct"]:
        for stat in [Stat.AVG, Stat.MEDIAN]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                metrics_dict[img_name][stat][metric] = []

        patients = ["489", "496", "499", "509", "513", "536", "538"]
        for patient in tqdm(patients):
            patient = "patient={}".format(patient)

            idl_img_dir = Dict()
            img_data = Dict()
            for idx in ["1", "2"]:
                idl_img_dir[idx] = os.path.join(
                    obs_study_dir[idx],
                    "patients",
                    patient,
                    "round=01",
                )

                if os.path.exists(idl_img_dir[idx]):
                    origin_pred = g.load_nii(
                        os.path.join(idl_img_dir[idx], "{}_pred.nii.gz".format(gtv)),
                        binary=True,
                    )
                    if img_name == "idl":
                        img_data[idx] = origin_pred
                    elif img_name == "correct":
                        correction_mask = g.load_nii(
                            os.path.join(
                                idl_img_dir[idx],
                                "{}_correction_mask.nii.gz".format(gtv),
                            ),
                            binary=True,
                        )
                        correction = g.load_nii(
                            os.path.join(
                                idl_img_dir[idx], "{}_correction.nii.gz".format(gtv)
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

            if img_data["1"] is None and img_data["2"] is None:
                dsc = 1.0
                msd = 0.0
                hd95 = 0.0
                apl_pct = 0.0
                apl_voxel = 0
                sdsc = 1.0

            elif img_data["1"] is not None and img_data["2"] is not None:
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

            else:
                g.error_exit("One of the observer has not patient data.")

            metrics_dict[img_name][patient][Metric.DSC] = dsc
            metrics_dict[img_name][patient][Metric.MSD] = msd
            metrics_dict[img_name][patient][Metric.HD95] = hd95
            metrics_dict[img_name][patient][Metric.APL_PCT] = apl_pct
            metrics_dict[img_name][patient][Metric.APL_VOXEL] = apl_voxel
            metrics_dict[img_name][patient][Metric.SDSC] = sdsc

            # record value for avg and median calculation
            for stat in [Stat.AVG, Stat.MEDIAN]:
                for metric in [
                    Metric.DSC,
                    Metric.MSD,
                    Metric.HD95,
                    Metric.APL_PCT,
                    Metric.APL_VOXEL,
                    Metric.SDSC,
                ]:
                    metrics_dict[img_name][stat][metric].append(
                        metrics_dict[img_name][patient][metric]
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
            metrics_dict[img_name][Stat.MEDIAN][metric] = g.calculate_median(
                metrics_dict[img_name][Stat.MEDIAN][metric]
            )
            metrics_dict[img_name][Stat.AVG][metric] = g.calculate_avg(
                metrics_dict[img_name][Stat.AVG][metric]
            )

    if os.path.exists(metrics_path):
        final_dict = g.load_json(metrics_path)
    else:
        final_dict = Dict()
    final_dict[gtv] = metrics_dict
    g.save_json(data=final_dict, path=metrics_path)


def plot_iov():
    obs_study_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_obs.study")
    for gtv in ["gtvt", "gtvn"]:
        if gtv == "gtvt":
            title_gtv = "GTVt"
        elif gtv == "gtvn":
            title_gtv = "GTVn"

        for img_name in tqdm(["idl", "correct"]):
            if img_name == "idl":
                title_img_name = "Initial Segmentation"
            elif img_name == "correct":
                title_img_name = "Final Correction"

            # Setting up the figure and axes for a 2x3 grid
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
            fig.suptitle(
                "Inter Observer Variation - {} {}".format(title_gtv, title_img_name),
                fontsize=20,
            )
            # Flattening the axes array for easier iteration
            axes = axes.flatten()

            i = 0
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                # Metric.APL_VOXEL,
                Metric.SDSC,
            ]:

                for observer_pair in [
                    "Jesper_vs_Kenneth",
                    "Kenneth_vs_Hanna",
                    "Hanna_vs_Jesper",
                ]:
                    json_path = os.path.join(
                        obs_study_dir, "iov_{}.json".format(observer_pair)
                    )
                    data = g.load_json(json_path)
                    data = data[gtv][img_name]["median"][metric]

                    if observer_pair == "Jesper_vs_Kenneth":
                        iov_12 = data
                    elif observer_pair == "Kenneth_vs_Hanna":
                        iov_23 = data
                    elif observer_pair == "Hanna_vs_Jesper":
                        iov_31 = data

                iov_matrix = np.array(
                    [[0, iov_12, iov_31], [iov_12, 0, iov_23], [iov_31, iov_23, 0]]
                )

                # Creating the heatmap with a white-to-blue color gradient
                sns.heatmap(
                    iov_matrix,
                    ax=axes[i],
                    annot=True,
                    cmap="Blues",
                    square=True,
                    cbar=False,
                )

                subtitle = explain_metric(metric)
                axes[i].set_title(subtitle)
                axes[i].set_xticklabels(["Observer 1", "Observer 2", "Observer 3"])
                axes[i].set_yticklabels(["Observer 1", "Observer 2", "Observer 3"])

                i += 1

            # turn off axis of the last figure
            # axes[-1].axis("off")
            fig.delaxes(axes[-1])

            fig_path = os.path.join(
                obs_study_dir, "iov_{}_{}.pdf".format(gtv, img_name)
            )
            plt.savefig(fig_path)


def seconds_to_minutes_decimal(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    decimal_minutes = remaining_seconds / 60
    total_minutes = minutes + decimal_minutes
    return total_minutes


def time_str_to_seconds(time_str: str):
    h, m, s = map(int, time_str.split(":"))
    seconds = h * 3600 + m * 60 + s
    return seconds


def plot_time_per_patient(obs_study_id_list: list):
    patients_list = ["489", "496", "499", "509", "513", "536", "538"]
    observers_list = ["Jesper", "Kenneth", "Hanna"]

    fig_data = Dict()
    for observer in observers_list:
        fig_data[observer] = []

    # loop through observer study train id
    for obs_study_id in tqdm(obs_study_id_list):
        if not obs_study_id.startswith("idl.gtvt_"):
            g.error_exit("Must be an 'idl.gtvt' id!")
        json_path = os.path.join(
            g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id, "time_used.json"
        )
        time_dict = g.load_json(json_path)

        # get observer name from train id
        for observer in observers_list:
            if observer in obs_study_id:
                break

        for patient in patients_list:
            total_gtvt_sec = 0
            total_gtvn_sec = 0
            patient_time = time_dict["patient={}".format(patient)]

            for i in [
                "click.gtvt.center",
                "draw.gtvt",
                "waiting.gtvt",
                "correct.gtvt",
            ]:
                cur_gtvt_sec = time_str_to_seconds(patient_time[i])
                total_gtvt_sec += cur_gtvt_sec

            for i in [
                "click.gtvt.center",
                "draw.gtvt",
                "click.gtvn.center",
                "waiting.gtvn",
                "correct.gtvn",
            ]:
                cur_gtvn_sec = time_str_to_seconds(patient_time[i])
                total_gtvn_sec += cur_gtvn_sec

            # print(total_gtvt_sec, total_gtvn_sec)
            fig_data[observer].append(max(total_gtvt_sec, total_gtvn_sec))

        # calculate avg and mean
        avg = g.calculate_avg(fig_data[observer])
        avg = round(avg)
        median = g.calculate_median(fig_data[observer])
        median = round(median)
        fig_data[observer].append(avg)
        fig_data[observer].append(median)

        for i in range(len(fig_data[observer])):
            fig_data[observer][i] = seconds_to_minutes_decimal(fig_data[observer][i])

    # Set up a 2x3 grid of subplots
    _, ax = plt.subplots(figsize=(8, 5))

    # Define bar width for clarity in grouped bars
    bar_width = 0.25

    # Calculate indices for x-axis where groups of bars will be located
    indices = np.arange(len(patients_list) + 2)

    # Plot bars for each observer
    for observer in observers_list:
        idx = fig_data.key_index(observer)
        ax.bar(
            x=indices + idx * bar_width,  # list
            height=fig_data[observer],  # list
            width=bar_width,
            label="Observer {}".format(observers_list.index(observer) + 1),
        )

    # Configure title and labels
    # ax.set_title("Initial Segmentation vs Final Correction")
    ax.set_xlabel("Patients")
    ax.set_ylabel("Time Used (Minutes)")
    ax.set_title("Time Used by Observers for Each Patient")

    # Set x-axis ticks to be centered under each group of bars
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "Mean", "Median"])

    # Add a legend to describe the observers
    # ax.legend()
    legend = ax.legend(loc="best")  # "upper right")
    legend.get_frame().set_alpha(0.3)

    # Adjust layout to prevent overlap and save the entire figure as a PDF
    plt.tight_layout()

    # Save the plot as a PDF file in the specified directory
    fig_path = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", "time_per_patient.pdf"
    )
    plt.savefig(fig_path, format="pdf")


def explain_idl_step(idl_step: str):
    if idl_step == "click.gtvn.center":
        return "Click GTVn Centers"
    elif idl_step == "click.gtvt.center":
        return "Click GTVt Center"
    elif idl_step == "correct.gtvn":
        return "Correct GTVn"
    elif idl_step == "correct.gtvt":
        return "Correct GTVt"
    elif idl_step == "draw.gtvt":
        return "Delineate GTVt Slices"
    elif idl_step == "waiting.gtvn":
        return "Generating GTVn Segmentation"
    elif idl_step == "waiting.gtvt":
        return "Generating GTVt Segmentation"


def plot_time_per_step(obs_study_id_list: list):
    observers_list = ["Jesper", "Kenneth", "Hanna"]
    idl_step_list = [
        "click.gtvt.center",
        "draw.gtvt",
        "waiting.gtvt",
        "correct.gtvt",
        "click.gtvn.center",
        "waiting.gtvn",
        "correct.gtvn",
    ]

    # Set up a 2x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Mean Time Consumption per iDL Step", fontsize=20)
    axes = axes.flatten()

    sub_fig_idx = 0
    # loop through observer study train id
    for obs_study_id in tqdm(obs_study_id_list):
        if not obs_study_id.startswith("idl.gtvt_"):
            g.error_exit("Must be an 'idl.gtvt' id!")

        # init data
        fig_data = Dict()
        for idl_step in idl_step_list:
            fig_data[idl_step]["value"] = []

        # load json
        json_path = os.path.join(
            g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id, "time_used.json"
        )
        time_dict = g.load_json(json_path)

        # calculate avrage time used of each step
        for patient in time_dict.keys():
            for idl_step in idl_step_list:
                seconds = time_str_to_seconds(time_dict[patient][idl_step])
                fig_data[idl_step]["value"].append(seconds)

            # fix gtvn correction time
            # sometimes gtvn corerction time is much lower than gtvt
            # this is caused by user's regret
            gtvt_time = (
                fig_data["waiting.gtvt"]["value"][-1]
                + fig_data["correct.gtvt"]["value"][-1]
            )
            gtvn_time = (
                fig_data["click.gtvn.center"]["value"][-1]
                + fig_data["waiting.gtvn"]["value"][-1]
                + fig_data["correct.gtvn"]["value"][-1]
            )
            if gtvt_time != gtvn_time:
                fig_data["correct.gtvn"]["value"][-1] = (
                    gtvt_time
                    - fig_data["click.gtvn.center"]["value"][-1]
                    - fig_data["waiting.gtvn"]["value"][-1]
                )

        for idl_step in idl_step_list:
            fig_data[idl_step]["value"] = g.calculate_avg(fig_data[idl_step]["value"])

        # add start time
        fig_data["click.gtvt.center"]["start"] = 0
        fig_data["draw.gtvt"]["start"] = fig_data["click.gtvt.center"]["value"]
        fig_data["waiting.gtvt"]["start"] = fig_data["click.gtvn.center"]["start"] = (
            fig_data["draw.gtvt"]["start"] + fig_data["draw.gtvt"]["value"]
        )
        fig_data["waiting.gtvn"]["start"] = (
            fig_data["click.gtvn.center"]["start"]
            + fig_data["click.gtvn.center"]["value"]
        )
        fig_data["correct.gtvt"]["start"] = (
            fig_data["waiting.gtvt"]["start"] + fig_data["waiting.gtvt"]["value"]
        )
        fig_data["correct.gtvn"]["start"] = (
            fig_data["waiting.gtvn"]["start"] + fig_data["waiting.gtvn"]["value"]
        )
        # time_range = (
        #     fig_data["correct.gtvt"]["start"] + fig_data["correct.gtvt"]["value"]
        # )
        # time_range *= 1.1
        # time_range = np.arange(time_range)

        # seconds to minutes
        for idl_step in idl_step_list:
            for i in ["start", "value"]:
                fig_data[idl_step][i] = seconds_to_minutes_decimal(
                    fig_data[idl_step][i]
                )

        # create sub fig
        ax = axes[sub_fig_idx]
        bar_height = 5
        step_space = 2  # Space between bars
        total_height = len(idl_step) * (bar_height + step_space)
        y_positions = [
            total_height - (i * (bar_height + step_space))
            for i in range(len(idl_step_list))
        ]

        # Creating a DataFrame for easier plotting
        # df_dict = Dict()
        # for idl_step in idl_step_list:
        #     lower = fig_data[idl_step]["start"]
        #     upper = fig_data[idl_step]["start"] + fig_data[idl_step]["value"]
        #     df_dict[explain_idl_step(idl_step)] = np.where(
        #         (time_range >= lower) & (time_range <= upper), 1, 0
        #     )
        # df_area = pd.DataFrame(df_dict, index=time_range)
        # df_area.plot.area(ax=ax, alpha=0.4)

        # set title
        for observer in observers_list:
            if observer in obs_study_id:
                break
        ax.set_title(
            "Observer {}".format(observers_list.index(observer) + 1),
        )
        ax.set_xlabel("Time (Minutes)")
        ax.set_yticks([y + bar_height / 2 for y in y_positions])
        ax.set_yticklabels([explain_idl_step(idl_step) for idl_step in idl_step_list])
        ax.grid(True)

        # Adding bars for each step
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
        ]

        for idl_step in idl_step_list:
            idx = idl_step_list.index(idl_step)
            lower = fig_data[idl_step]["start"]
            value = fig_data[idl_step]["value"]
            ax.broken_barh(
                [(lower, value)],
                (y_positions[idx], bar_height),
                facecolors=colors[idx],
            )

        # next sub plot
        sub_fig_idx += 1

    # Adjust layout to prevent overlap and save the entire figure as a PDF
    plt.tight_layout()
    # Save the plot as a PDF file in the specified directory
    fig_path = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", "time_per_step.pdf"
    )
    plt.savefig(fig_path, format="pdf")
