import os

import numpy as np
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Json, Nii
from segment_metric import avg_surface_distance_symmetric, dice, hausdorff_distance_95
from tqdm import tqdm


def cal_obs_study_metrics(obs_study_id: str):
    baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_obs.study", "baseline")

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
    obs_metrics_path = os.path.join(obs_study_dir, "obs_metrics.json")
    Json.save(data=obs_metrics, path=obs_metrics_path)

    patient_dir_list = Dir.get_sub_dirs(os.path.join(obs_study_dir, "patients"))

    for patient_dir in tqdm(patient_dir_list):
        if not os.path.exists(os.path.join(baseline_dir, "patients", patient_dir)):
            continue

        baseline_pred = Nii.load(
            os.path.join(
                baseline_dir, "patients", patient_dir, "{}_pred.nii.gz".format(gtv)
            ),
            binary=True,
        )
        idl_img_dir = os.path.join(obs_study_dir, "patients", patient_dir, "round=01")
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

        volume_pairs = Dict()
        volume_pairs["correct.pred"] = (final_pred, origin_pred)
        volume_pairs["pred.baseline"] = (final_pred, baseline_pred)
        volume_pairs["correct.baseline"] = (final_pred, baseline_pred)

        for cur_key in volume_pairs.keys():
            volume_1 = volume_pairs[cur_key][0]
            volume_2 = volume_pairs[cur_key][1]
            dsc = dice(
                test=volume_1,
                reference=volume_2,
                nan_for_nonexisting=False,
            )

            msd = avg_surface_distance_symmetric(
                test=volume_1,
                reference=volume_2,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

            hd95 = hausdorff_distance_95(
                test=volume_1,
                reference=volume_2,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

            obs_metrics[patient_dir][cur_key]["dsc"] = dsc
            obs_metrics[patient_dir][cur_key]["msd"] = msd
            obs_metrics[patient_dir][cur_key]["hd95"] = hd95

    Json.save(data=obs_metrics, path=obs_metrics_path)


obs_study_id_list = [
    "idl.gtvt_2024.03.18.09.05.54_Jesper",
    "idl.gtvn_2024.03.18.09.05.54_Jesper",
    "idl.gtvt_2024.03.21.13.07.10_Hanna",
    "idl.gtvn_2024.03.21.13.07.10_Hanna",
    "idl.gtvt_2024.03.22.10.08.33_Kenneth",
    "idl.gtvn_2024.03.22.10.08.33_Kenneth",
]
for obs_study_id in obs_study_id_list:
    cal_obs_study_metrics(obs_study_id)
