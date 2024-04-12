import os

from added_path_length import APL
from custom import Debug, Dict, Dir
from custom import Global as g
from custom import Json, Nii, Value
from segment_metric import avg_surface_distance_symmetric, dice, hausdorff_distance_95
from str_lib import Metric, Stat
from surface_dice import SurfaceDice
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

    for stat in [Stat.AVG, Stat.MEDIAN]:
        for i in ["correct.vs.origin"]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                obs_metrics[stat][i][metric] = []

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
        volume_pairs["correct.vs.origin"] = (final_pred, origin_pred)
        # volume_pairs["origin.vs.baseline"] = (origin_pred, baseline_pred)
        # volume_pairs["correct.vs.baseline"] = (final_pred, baseline_pred)

        for cur_key in volume_pairs.keys():
            reference = volume_pairs[cur_key][0]
            test = volume_pairs[cur_key][1]
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

            obs_metrics[patient_dir][cur_key][Metric.DSC] = dsc
            obs_metrics[patient_dir][cur_key][Metric.MSD] = msd
            obs_metrics[patient_dir][cur_key][Metric.HD95] = hd95

            # added path length
            apl = APL(reference_structure=reference, other_structure=test)
            obs_metrics[patient_dir][cur_key][Metric.APL_PCT] = apl.get_apl(
                normalized=True
            )
            obs_metrics[patient_dir][cur_key][Metric.APL_VOXEL] = apl.get_apl(
                normalized=False
            )

            # surface dice
            sdsc = SurfaceDice(reference_image=reference, other_image=test)
            obs_metrics[patient_dir][cur_key][Metric.SDSC] = sdsc.get_surface_dice()

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
                    obs_metrics[stat][cur_key][metric].append(
                        obs_metrics[patient_dir][cur_key][metric]
                    )

    for i in ["correct.vs.origin"]:
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


obs_study_id_list = [
    # "idl.gtvt_2024.03.18.09.05.54_Jesper",
    # "idl.gtvn_2024.03.18.09.05.54_Jesper",
    # "idl.gtvt_2024.03.21.13.07.10_Hanna",
    # "idl.gtvn_2024.03.21.13.07.10_Hanna",
    # "idl.gtvt_2024.03.22.10.08.33_Kenneth",
    # "idl.gtvn_2024.03.22.10.08.33_Kenneth",
    "idl.gtvn_2024.04.12.12.05.44_Kenneth",
    "idl.gtvt_2024.04.12.12.05.44_Kenneth",
]
for obs_study_id in obs_study_id_list:
    cal_obs_study_metrics(obs_study_id)
