import csv
import os

import custom as g
import cv2
import numpy as np
import torch
from added_path_len import APL
from custom_dict import Dict
from monai.metrics import compute_surface_dice
from segment_metric import avg_surface_distance_symmetric, dice, hausdorff_distance_95

# from skimage import morphology
from str_lib import Metric, ObsStudyStep, Plane, Stat

# from surface_dice import SurfaceDice
from tqdm import tqdm


def cal_obs_study_metrics_3d(obs_study_id: str):
    # baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_obs.study", "baseline")

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
    metrics_path = os.path.join(obs_study_dir, "obs_study_metrics_3d.json")
    for stat in [Stat.AVG, Stat.MEDIAN]:
        for i in ["correct.vs.idl"]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                metrics_dict[stat][i][metric] = []
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
        # # skip the patients without baseline
        # if not os.path.exists(os.path.join(baseline_dir, "patients", patient)):
        #     continue
        # baseline_pred = g.load_nii(
        #     os.path.join(
        #         baseline_dir, "patients", patient, "{}_pred.nii.gz".format(gtv)
        #     ),
        #     binary=True,
        # )

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

        volume_pairs = Dict()
        volume_pairs["correct.vs.idl"] = (final_pred, origin_pred)
        # volume_pairs["idl.vs.baseline"] = (origin_pred, baseline_pred)
        # volume_pairs["correct.vs.baseline"] = (final_pred, baseline_pred)

        for i in volume_pairs.keys():
            reference = volume_pairs[i][0]
            test = volume_pairs[i][1]

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

            metrics_dict[patient][i][Metric.DSC] = dsc
            metrics_dict[patient][i][Metric.MSD] = msd
            metrics_dict[patient][i][Metric.HD95] = hd95

            # added path length
            if os.path.exists(patient_dir):
                apl = APL(reference_structure=reference, other_structure=test)
                apl_pct = apl.get_apl(normalized=True)
                apl_voxel = apl.get_apl(normalized=False)
            else:
                apl_pct = 0.0
                apl_voxel = 0

            metrics_dict[patient][i][Metric.APL_PCT] = apl_pct
            metrics_dict[patient][i][Metric.APL_VOXEL] = apl_voxel

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
                metrics_dict[patient][i][Metric.SDSC] = sdsc.item()
                # sdsc = SurfaceDice(reference_image=reference, other_image=test)
                # metrics_dict[patient][i][Metric.SDSC] = sdsc.get_surface_dice()
            else:
                metrics_dict[patient][i][Metric.SDSC] = 1.0

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
                    metrics_dict[stat][i][metric].append(
                        metrics_dict[patient][i][metric]
                    )

    for i in ["correct.vs.idl"]:
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            metrics_dict[Stat.MEDIAN][i][metric] = g.calculate_median(
                metrics_dict[Stat.MEDIAN][i][metric]
            )
            metrics_dict[Stat.AVG][i][metric] = g.calculate_avg(
                metrics_dict[Stat.AVG][i][metric]
            )

    g.save_json(data=metrics_dict, path=metrics_path)


def cal_obs_study_metrics_gtvt_central_slices(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        g.error_exit("Must be an 'idl.gtvt' id")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    metrics_dict = Dict()
    metrics_path = os.path.join(
        obs_study_dir, "obs_study_metrics_gtvt_central_slices.json"
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

            # g.save_nii(delineation_2d, os.path.join(g.DEBUG_DIR, "before_open.nii.gz"))
            kernel = np.ones((3, 3), np.uint8)
            delineation_2d = cv2.morphologyEx(delineation_2d, cv2.MORPH_OPEN, kernel)
            # g.save_nii(delineation_2d, os.path.join(g.DEBUG_DIR, "after_open.nii.gz"))

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
            # sdsc = SurfaceDice(
            #     reference_image=final_pred_2d, other_image=delineation_2d
            # )
            # metrics_dict[patient][Metric.SDSC][plane] = sdsc.get_surface_dice()

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


def create_table_metrics_3d(obs_study_id_list: list):
    table_path = Dict()
    table_data = Dict()

    for i in ["gtvt", "gtvn"]:
        table_path[i] = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            "obs_study_metrics_3d_{}.csv".format(i),
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
            for stat in [Stat.AVG, Stat.MEDIAN]:
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
            os.path.join(obs_study_dir, "obs_study_metrics_3d.json")
        )

        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            for stat in [Stat.AVG, Stat.MEDIAN]:
                cur_value = metrics_dict[stat]["correct.vs.idl"][metric]
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


def create_table_metrics_gtvt_central_slices(obs_study_id_list: list):
    # tabel_path=Dict()

    table_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        "obs_study_metrics_gtvt_central_slices.csv",
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
        Metric.SDSC,
    ]:
        for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
            for stat in [Stat.AVG, Stat.MEDIAN]:
                cur_item = [metric, plane, stat, "", "", ""]
                table_data.append(cur_item)

    for obs_study_id in tqdm(obs_study_id_list):
        if not obs_study_id.startswith("idl.gtvt_"):
            g.error_exit("Must be an 'idl.gtvt' id")

        obs_study_dir = os.path.join(
            g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
        )
        metrics_dict = g.load_json(
            os.path.join(obs_study_dir, "obs_study_metrics_gtvt_central_slices.json")
        )

        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            for plane in [Plane.TRANSVERSE, Plane.CORONAL, Plane.SAGITTAL]:
                for stat in [Stat.AVG, Stat.MEDIAN]:
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


if 1:
    obs_study_id_list = [
        "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
        "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvn_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvn_2024.04.18.11.04.48_Hanna_research",
    ]
    # for obs_study_id in obs_study_id_list:
    #     cal_obs_study_metrics_3d(obs_study_id)

    create_table_metrics_3d(obs_study_id_list)

if 1:
    obs_study_id_list = [
        "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
    ]
    # for obs_study_id in obs_study_id_list:
    #     cal_obs_study_metrics_gtvt_central_slices(obs_study_id)

    create_table_metrics_gtvt_central_slices(obs_study_id_list)
