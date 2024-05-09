import csv
import os

import cv2
import global_core as g
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from added_path_len import APL
from custom_dict import Dict
from segment_metric import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
    surface_dice,
    surface_distances,
)
from str_lib import Metric, ObsStudyStep, Plane, Stat
from tqdm import tqdm


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
                "3d_idl_vs_correct_{}_{}.pdf".format(gtv, metric),
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
                    "3d_idl_vs_correct.json",
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
            ax.set_title("Initial Segmentation vs Final Correction")
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

                if patient == "patient=513":
                    g.save_nii(
                        delineation_2d,
                        os.path.join(
                            g.DEBUG_DIR,
                            "{}_{}_delineation.nii.gz".format(patient, plane),
                        ),
                    )
                    g.save_nii(
                        origin_pred_2d,
                        os.path.join(
                            g.DEBUG_DIR,
                            "{}_{}_idl.nii.gz".format(patient, plane),
                        ),
                    )
                    g.save_nii(
                        final_pred_2d,
                        os.path.join(
                            g.DEBUG_DIR,
                            "{}_{}_correct.nii.gz".format(patient, plane),
                        ),
                    )

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

        # calculate avg and median
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
        ]:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
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
            # hd95 = hausdorff_distance_95(
            #     test=delineation_2d,
            #     reference=delineation_2d_fixed,
            #     none_for_nonexisting=True,
            #     voxel_spacing=(1, 1),  # g.NII_SPACING,
            # )

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
        metrics_dict[patient] = hd100

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

        for metric in [Metric.MSD, Metric.HD95]:
            # init fig_path and fig_data
            fig_path = os.path.join(
                g.TRAIN_RESULTS_DIR,
                "baseline_obs.study",
                "2d_{}_vs_{}_{}.pdf".format(target_1, target_2, metric),
            )

            # Create a grouped bar plot
            _, ax = plt.subplots()

            for obs_study_id in tqdm(obs_study_id_list):

                x_list = []
                y_list = []

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
                    # x_value = max(
                    #     x_value[Plane.TRANSVERSE],
                    #     x_value[Plane.CORONAL],
                    #     x_value[Plane.SAGITTAL],
                    # )
                    if 1:
                        x_list.append(x_value)
                    else:
                        for i in range(3):
                            x_list.append(x_value)

                    y_value = y_axis_dict[patient][metric]
                    if 1:
                        y_value = max(
                            y_value[Plane.TRANSVERSE],
                            y_value[Plane.CORONAL],
                            y_value[Plane.SAGITTAL],
                        )
                        y_list.append(y_value)
                    else:
                        y_list.append(y_value[Plane.TRANSVERSE])
                        y_list.append(y_value[Plane.CORONAL])
                        y_list.append(y_value[Plane.SAGITTAL])

                ax.scatter(
                    x=x_list,
                    y=y_list,
                    label=observer,
                )

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

            ax.set_title("GTVt - {} vs {}".format(str_1, str_2))

            ax.set_ylabel(
                ylabel=metric.upper(),
                rotation=0,
                position=(0, 1),
                va="bottom",
                labelpad=-9,
            )
            ax.set_xlabel(
                xlabel="Anatomical Plane Variation (HD 100) of GTVt User Input",
            )

            # Add a legend to describe the observers
            ax.legend()

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

    for img_name in ["idl", "correct"]:
        metrics_dict = Dict()
        metrics_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "iov_{}_vs_{}_{}_{}.json".format(
                observer["1"], observer["2"], gtv, img_name
            ),
        )

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

            metrics_dict[patient][Metric.DSC] = dsc
            metrics_dict[patient][Metric.MSD] = msd
            metrics_dict[patient][Metric.HD95] = hd95
            metrics_dict[patient][Metric.APL_PCT] = apl_pct
            metrics_dict[patient][Metric.APL_VOXEL] = apl_voxel
            metrics_dict[patient][Metric.SDSC] = sdsc

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
            metrics_dict[Stat.AVG][metric] = g.calculate_avg(
                metrics_dict[Stat.AVG][metric]
            )

        g.save_json(data=metrics_dict, path=metrics_path)


def plot_iov():

    # Assuming you have your data
    # Generate random data for demonstration purposes
    np.random.seed(42)
    data_A = np.random.normal(20, 3, 10)
    data_B = np.random.normal(22, 3, 10)
    data_C = np.random.normal(21, 3, 10)

    # Compute the correlation matrix
    data = np.array([data_A, data_B, data_C])
    corr_matrix = np.corrcoef(data)

    # Creating the heatmap with a white-to-blue color gradient
    plt.figure(figsize=(8, 6))  # Adjust the size to fit your needs
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="Blues",
        xticklabels=["Observer A", "Observer B", "Observer C"],
        yticklabels=["Observer A", "Observer B", "Observer C"],
    )
    plt.title("Heatmap of Correlation Among Observers")

    plt.savefig(os.path.join(g.DEBUG_DIR, "heat_map.pdf"))
