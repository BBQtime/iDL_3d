import os

import cv2
import numpy as np
import torch
from added_path_length import APL
from custom import Debug, Dict
from custom import Global as g
from custom import Json, Nii, Value
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
        Debug.error_exit("obs study train id error")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    obs_metrics = Dict()
    obs_metrics_path = os.path.join(obs_study_dir, "obs_study_metrics_3d.json")
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
                obs_metrics[stat][i][metric] = []
    Json.save(data=obs_metrics, path=obs_metrics_path)

    patient_list = []
    # open "obs_study_step.json" and find approved patients
    obs_study_step_json_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        Value.replace_char(obs_study_id, 7, "t"),
        "obs_study_step.json",
    )
    obs_study_step = Json.load(obs_study_step_json_path)
    for patient in obs_study_step.keys():
        if obs_study_step[patient] == ObsStudyStep.APPROVED:
            patient_list.append(patient)

    for patient in tqdm(patient_list):
        # # skip the patients without baseline
        # if not os.path.exists(os.path.join(baseline_dir, "patients", patient)):
        #     continue
        # baseline_pred = Nii.load(
        #     os.path.join(
        #         baseline_dir, "patients", patient, "{}_pred.nii.gz".format(gtv)
        #     ),
        #     binary=True,
        # )

        patient_dir = os.path.join(obs_study_dir, "patients", patient)

        if os.path.exists(patient_dir):
            idl_img_dir = os.path.join(patient_dir, "round=01")
            origin_pred = Nii.load(
                os.path.join(idl_img_dir, "{}_pred.nii.gz".format(gtv)),
                binary=True,
            )
            correction_mask = Nii.load(
                os.path.join(idl_img_dir, "{}_correction_mask.nii.gz".format(gtv)),
                binary=True,
            )
            correction = Nii.load(
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

            obs_metrics[patient][i][Metric.DSC] = dsc
            obs_metrics[patient][i][Metric.MSD] = msd
            obs_metrics[patient][i][Metric.HD95] = hd95

            # added path length
            if os.path.exists(patient_dir):
                apl = APL(reference_structure=reference, other_structure=test)
                apl_pct = apl.get_apl(normalized=True)
                apl_voxel = apl.get_apl(normalized=False)
            else:
                apl_pct = 0.0
                apl_voxel = 0

            obs_metrics[patient][i][Metric.APL_PCT] = apl_pct
            obs_metrics[patient][i][Metric.APL_VOXEL] = apl_voxel

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
                obs_metrics[patient][i][Metric.SDSC] = sdsc.item()
                # sdsc = SurfaceDice(reference_image=reference, other_image=test)
                # obs_metrics[patient][i][Metric.SDSC] = sdsc.get_surface_dice()
            else:
                obs_metrics[patient][i][Metric.SDSC] = 1.0

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
                    obs_metrics[stat][i][metric].append(obs_metrics[patient][i][metric])

    for i in ["correct.vs.idl"]:
        for metric in [
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
            Metric.APL_PCT,
            Metric.APL_VOXEL,
            Metric.SDSC,
        ]:
            obs_metrics[Stat.MEDIAN][i][metric] = Value.median(
                obs_metrics[Stat.MEDIAN][i][metric]
            )
            obs_metrics[Stat.AVG][i][metric] = Value.avg(
                obs_metrics[Stat.AVG][i][metric]
            )

    Json.save(data=obs_metrics, path=obs_metrics_path)


def cal_obs_study_metrics_gtvt_central_slices(obs_study_id: str):
    if not obs_study_id.startswith("idl.gtvt_"):
        Debug.error_exit("Must be an 'idl.gtvt' id")

    obs_study_dir = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
    )

    obs_metrics = Dict()
    obs_metrics_path = os.path.join(
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
                obs_metrics[stat][metric][plane] = []
    Json.save(data=obs_metrics, path=obs_metrics_path)

    patient_list = []
    # open "obs_study_step.json" and find approved patients
    obs_study_step = Json.load(
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

        delineation = Nii.load(
            os.path.join(idl_img_dir, "gtvt_delineation.nii.gz"),
            binary=True,
        )
        origin_pred = Nii.load(
            os.path.join(idl_img_dir, "gtvt_pred.nii.gz"),
            binary=True,
        )
        correction_mask = Nii.load(
            os.path.join(idl_img_dir, "gtvt_correction_mask.nii.gz"),
            binary=True,
        )
        correction = Nii.load(
            os.path.join(idl_img_dir, "gtvt_correction.nii.gz"),
            binary=True,
        )
        final_pred = g.combine_pred_correction(
            origin_pred=origin_pred,
            correction=correction,
            correction_mask=correction_mask,
        )

        selected_slices = Json.load(os.path.join(patient_dir, "selected_slices.json"))

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

            # Nii.save(delineation_2d, os.path.join(g.DEBUG_DIR, "before_open.nii.gz"))
            kernel = np.ones((3, 3), np.uint8)
            delineation_2d = cv2.morphologyEx(delineation_2d, cv2.MORPH_OPEN, kernel)
            # Nii.save(delineation_2d, os.path.join(g.DEBUG_DIR, "after_open.nii.gz"))

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
            obs_metrics[patient][Metric.DSC][plane] = dsc
            obs_metrics[patient][Metric.MSD][plane] = msd
            obs_metrics[patient][Metric.HD95][plane] = hd95

            # added path length
            apl = APL(reference_structure=final_pred_2d, other_structure=delineation_2d)
            apl_pct = apl.get_apl(normalized=True)
            apl_voxel = apl.get_apl(normalized=False)
            obs_metrics[patient][Metric.APL_PCT][plane] = apl_pct
            obs_metrics[patient][Metric.APL_VOXEL][plane] = apl_voxel

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
            obs_metrics[patient][Metric.SDSC][plane] = sdsc.item()
            # sdsc = SurfaceDice(
            #     reference_image=final_pred_2d, other_image=delineation_2d
            # )
            # obs_metrics[patient][Metric.SDSC][plane] = sdsc.get_surface_dice()

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
                    obs_metrics[stat][metric][plane].append(
                        obs_metrics[patient][metric][plane]
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
            obs_metrics[Stat.MEDIAN][metric][plane] = Value.median(
                obs_metrics[Stat.MEDIAN][metric][plane]
            )
            obs_metrics[Stat.AVG][metric][plane] = Value.avg(
                obs_metrics[Stat.AVG][metric][plane]
            )

    Json.save(data=obs_metrics, path=obs_metrics_path)


if 1:
    obs_study_id_list = [
        "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
        "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvn_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvn_2024.04.18.11.04.48_Hanna_research",
    ]
    for obs_study_id in obs_study_id_list:
        cal_obs_study_metrics_3d(obs_study_id)

else:
    obs_study_id_list = [
        "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
    ]
    for obs_study_id in obs_study_id_list:
        cal_obs_study_metrics_gtvt_central_slices(obs_study_id)
