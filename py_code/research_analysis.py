import csv
import os

import cv2
import global_core as g
import matplotlib.pyplot as plt
import numpy as np
import torch
from added_path_len import APL
from custom_dict import Dict
from monai.metrics import compute_surface_dice
from segment_metric import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance,
    hausdorff_distance_95,
)
from str_lib import Metric, ObsStudyStep, Plane, Stat
from tqdm import tqdm


def calculate_metrics_correct_vs_idl(obs_study_id: str):

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
    metrics_path = os.path.join(obs_study_dir, "correct_vs_idl.json")
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
            test = np.expand_dims(test, axis=0)
            test = np.expand_dims(test, axis=0)
            if len(test.shape) == 5:
                test = np.transpose(test, (0, 1, 3, 4, 2))
            reference = np.expand_dims(reference, axis=0)
            reference = np.expand_dims(reference, axis=0)
            if len(reference.shape) == 5:
                reference = np.transpose(reference, (0, 1, 3, 4, 2))
            sdsc = compute_surface_dice(
                y_pred=torch.tensor(test),
                y=torch.tensor(reference),
                class_thresholds=[1.0],
            )
            # tensor to float
            metrics_dict[patient][Metric.SDSC] = sdsc.item()
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


def calculate_metrics_gtvt_delineation_vs_idl(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        g.error_exit("Must be an 'idl.gtvt' id")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    metrics_dict = Dict()
    metrics_path = os.path.join(obs_study_dir, "gtvt_delineation_vs_idl.json")
    for stat in [Stat.AVG, Stat.MEDIAN]:
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
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
    # nii_idx = 1
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

        selected_slices = g.load_json(os.path.join(patient_dir, "selected_slices.json"))

        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            slice_id = int(selected_slices[plane]["round=01"])

            if plane == Plane.TRANSVERSE:
                delineation_2d = delineation[slice_id, :, :]
                final_pred_2d = final_pred[slice_id, :, :]
            elif plane == Plane.CORONAL:
                delineation_2d = delineation[:, slice_id, :]
                final_pred_2d = final_pred[:, slice_id, :]
            elif plane == Plane.SAGITTAL:
                delineation_2d = delineation[:, :, slice_id]
                final_pred_2d = final_pred[:, :, slice_id]

            # g.save_nii(
            #     delineation_2d,
            #     os.path.join(g.DEBUG_DIR, "{:02d}_before_open.nii.gz".format(nii_idx)),
            # )
            kernel = np.ones((3, 3), np.uint8)
            delineation_2d = cv2.morphologyEx(delineation_2d, cv2.MORPH_OPEN, kernel)
            delineation_2d = cv2.morphologyEx(delineation_2d, cv2.MORPH_CLOSE, kernel)
            # g.save_nii(
            #     delineation_2d,
            #     os.path.join(g.DEBUG_DIR, "{:02d}_after_open.nii.gz".format(nii_idx)),
            # )
            # nii_idx += 1

            # dsc/msd/hd95
            dsc = dice(
                test=delineation_2d,
                reference=final_pred_2d,
                nan_for_nonexisting=False,
            )
            msd = avg_surface_distance_symmetric(
                test=delineation_2d,
                reference=final_pred_2d,
                none_for_nonexisting=True,
                voxel_spacing=(1, 1),  # g.NII_SPACING,
            )
            hd95 = hausdorff_distance_95(
                test=delineation_2d,
                reference=final_pred_2d,
                none_for_nonexisting=True,
                voxel_spacing=(1, 1),  # g.NII_SPACING,
            )

            metrics_dict[patient][Metric.DSC][plane] = dsc
            metrics_dict[patient][Metric.MSD][plane] = msd
            metrics_dict[patient][Metric.HD95][plane] = hd95

            # added path length
            apl = APL(reference_structure=final_pred_2d, other_structure=delineation_2d)
            apl_pct = apl.get_apl(normalized=True)
            apl_voxel = apl.get_apl(normalized=False)
            metrics_dict[patient][Metric.APL_PCT][plane] = apl_pct
            metrics_dict[patient][Metric.APL_VOXEL][plane] = apl_voxel

            # surface dice
            final_pred_2d = np.expand_dims(final_pred_2d, axis=0)
            final_pred_2d = np.expand_dims(final_pred_2d, axis=0)
            if len(final_pred_2d.shape) == 5:
                final_pred_2d = np.transpose(final_pred_2d, (0, 1, 3, 4, 2))
            delineation_2d = np.expand_dims(delineation_2d, axis=0)
            delineation_2d = np.expand_dims(delineation_2d, axis=0)
            if len(delineation_2d.shape) == 5:
                delineation_2d = np.transpose(delineation_2d, (0, 1, 3, 4, 2))
            sdsc = compute_surface_dice(
                y_pred=torch.tensor(final_pred_2d),
                y=torch.tensor(delineation_2d),
                class_thresholds=[1.0],
            )
            # tensor to float
            metrics_dict[patient][Metric.SDSC][plane] = sdsc.item()

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
                    metrics_dict[stat][metric][plane].append(
                        metrics_dict[patient][metric][plane]
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
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            metrics_dict[Stat.MEDIAN][metric][plane] = g.calculate_median(
                metrics_dict[Stat.MEDIAN][metric][plane]
            )
            metrics_dict[Stat.AVG][metric][plane] = g.calculate_avg(
                metrics_dict[Stat.AVG][metric][plane]
            )

    g.save_json(data=metrics_dict, path=metrics_path)


def create_table_correct_vs_idl(obs_study_id_list: list):
    table_path = Dict()
    table_data = Dict()

    for i in ["gtvt", "gtvn"]:
        table_path[i] = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "correct_vs_idl_{}.csv".format(i),
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
        metrics_dict = g.load_json(os.path.join(obs_study_dir, "correct_vs_idl.json"))

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


def create_table_gtvt_delineation_vs_idl(obs_study_id_list: list):

    table_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        "gtvt_delineation_vs_idl.csv",
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
            g.error_exit("Must be an 'idl.gtvt' id")

        obs_study_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
        )
        metrics_dict = g.load_json(
            os.path.join(obs_study_dir, "gtvt_delineation_vs_idl.json")
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
                        if item[0] == metric and item[1] == plane and item[2] == stat:
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


def plot_fig_correct_vs_idl(obs_study_id_list: list):

    patients_list = ["489", "496", "499", "509", "513", "536", "538"]
    observers_list = ["Jesper", "Kenneth", "Hanna"]

    for metric in tqdm(
        [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]
    ):
        # init fig_path and fig_data
        fig_path = Dict()
        fig_data = Dict()

        for gtv in ["gtvt", "gtvn"]:
            fig_path[gtv] = os.path.join(
                g.TRAIN_RESULTS_DIR,
                "baseline_obs.study",
                "correct_vs_idl_{}_{}.pdf".format(gtv, metric),
            )

            for observer in observers_list:
                fig_data[gtv][observer] = []

        # loop through observer study train id
        for obs_study_id in obs_study_id_list:
            if obs_study_id.startswith("idl.gtvn_"):
                gtv = "gtvn"
            elif obs_study_id.startswith("idl.gtvt_"):
                gtv = "gtvt"
            else:
                g.error_exit("obs study train id error")

            metrics_dict = g.load_json(
                os.path.join(
                    g.TRAIN_RESULTS_DIR,
                    "baseline_obs.study",
                    obs_study_id,
                    "correct_vs_idl.json",
                )
            )

            # get observer name from train id
            for observer in observers_list:
                if observer in obs_study_id:
                    break

            for patient in patients_list:
                fig_data[gtv][observer].append(
                    metrics_dict["patient={}".format(patient)][metric]
                )

        for gtv in ["gtvt", "gtvn"]:
            # Create a grouped bar plot
            _, ax = plt.subplots()

            # Define bar width for clarity in grouped bars
            bar_width = 0.25

            # Calculate indices for x-axis where groups of bars will be located
            indices = np.arange(len(patients_list))

            # Plot bars for each observer
            for observer in observers_list:
                idx = fig_data[gtv].key_index(observer)
                ax.bar(
                    x=indices + idx * bar_width,  # list
                    height=fig_data[gtv][observer],  # list
                    width=bar_width,
                    label="Observer {}".format(observers_list.index(observer) + 1),
                )

            # Configure title and labels
            ax.set_title("Initial Segmentation vs Final Segmentation")
            # if metric==Metric.
            #     ylabel=
            ax.set_ylabel(
                ylabel=metric.upper(),
                rotation=0,
                position=(0, 1),
                va="bottom",
                labelpad=-9,
            )
            ax.set_xlabel(
                xlabel="Patient",
                position=(1, 0),
                labelpad=-11,
            )

            # Define y-axis range to accommodate label placement above 1.0
            # Define y-axis ticks to display key points including 1.0
            if metric == Metric.DSC or metric == Metric.SDSC:
                ax.set_ylim(0, 1.2)
                # ax.set_yticks(np.linspace(0, 1, 6))

            # Set x-axis ticks to be centered under each group of bars
            ax.set_xticks(indices + bar_width)
            ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])

            # Add a legend to describe the observers
            ax.legend()

            # Save the plot as a PDF file in the specified directory
            plt.savefig(fig_path[gtv], format="pdf")


def calculate_gtvt_delineation_hd100(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        g.error_exit("Must be an 'idl.gtvt' id")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    metrics_dict = Dict()
    metrics_path = os.path.join(obs_study_dir, "gtvt_delineation_hd100.json")
    for stat in [Stat.AVG, Stat.MEDIAN]:
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            metrics_dict[stat][plane] = []
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

            hd100 = hausdorff_distance(
                test=delineation_2d,
                reference=delineation_2d_fixed,
                none_for_nonexisting=True,
                voxel_spacing=(1, 1),  # g.NII_SPACING,
            )
            metrics_dict[patient][plane] = hd100

            # record value for avg and median calculation
            for stat in [Stat.AVG, Stat.MEDIAN]:
                metrics_dict[stat][plane].append(hd100)

    # calculate avg and median
    for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
        metrics_dict[Stat.MEDIAN][plane] = g.calculate_median(
            metrics_dict[Stat.MEDIAN][plane]
        )
        metrics_dict[Stat.AVG][plane] = g.calculate_avg(metrics_dict[Stat.AVG][plane])

    g.save_json(data=metrics_dict, path=metrics_path)


def plot_fig_gtvt_delineation_hd100(obs_study_id_list: list):

    for metric in [Metric.MSD, Metric.HD95]:
        # init fig_path and fig_data
        fig_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "gtvt_delineation_hd100_{}.pdf".format(metric),
        )

        fig_data = Dict()
        fig_data["x"] = []
        fig_data["y"] = []

        for obs_study_id in tqdm(obs_study_id_list):
            if not obs_study_id.startswith("idl.gtvt_"):
                g.error_exit("Must be an 'idl.gtvt' id")

            obs_study_dir = os.path.join(
                g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
            )
            x_axis_dict = g.load_json(
                os.path.join(obs_study_dir, "gtvt_delineation_hd100.json")
            )
            y_axis_dict = g.load_json(
                os.path.join(obs_study_dir, "gtvt_delineation_vs_idl.json")
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
                x_value = max(
                    x_value[Plane.TRANSVERSE],
                    x_value[Plane.CORONAL],
                    x_value[Plane.SAGITTAL],
                )
                fig_data["x"].append(x_value)

                y_value = y_axis_dict[patient][metric]
                y_value = max(
                    y_value[Plane.TRANSVERSE],
                    y_value[Plane.CORONAL],
                    y_value[Plane.SAGITTAL],
                )
                fig_data["y"].append(y_value)

        # Create a grouped bar plot
        _, ax = plt.subplots()

        # Generate random colors
        # colors = [g.random_color() for _ in range(len(fig_data["x"]))]

        ax.scatter(
            x=fig_data["x"],
            y=fig_data["y"],
            # color=colors,
            # width=0.4,
        )

        # Configure title and labels
        ax.set_title("GTVt User Input vs Initial Segmentation")
        ax.set_ylabel(
            ylabel=metric.upper(),
            rotation=0,
            position=(0, 1),
            va="bottom",
            labelpad=-9,
        )
        ax.set_xlabel(
            xlabel="Max Anatomical Variation (Hausdorff Distance) of GTVt User Input",
            # position=(1, 0),
            # labelpad=-11,
        )

        # Add a legend to describe the observers
        ax.legend()

        # Save the plot as a PDF file in the specified directory
        plt.savefig(fig_path, format="pdf")
