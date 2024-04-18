import os

from added_path_length import APL
from custom import Debug, Dict
from custom import Global as g
from custom import Json, Nii, Value
from segment_metric import avg_surface_distance_symmetric, dice, hausdorff_distance_95
from str_lib import Metric, ObsStudyStep, Stat
from surface_dice import SurfaceDice
from tqdm import tqdm


def cal_obs_study_metrics(obs_study_id: str):
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
    obs_metrics_path = os.path.join(obs_study_dir, "obs_study_metrics.json")
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
        volume_pairs["correct.vs.origin"] = (final_pred, origin_pred)
        # volume_pairs["origin.vs.baseline"] = (origin_pred, baseline_pred)
        # volume_pairs["correct.vs.baseline"] = (final_pred, baseline_pred)

        for cur_key in volume_pairs.keys():
            reference = volume_pairs[cur_key][0]
            test = volume_pairs[cur_key][1]

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

            obs_metrics[patient][cur_key][Metric.DSC] = dsc
            obs_metrics[patient][cur_key][Metric.MSD] = msd
            obs_metrics[patient][cur_key][Metric.HD95] = hd95

            # added path length
            if os.path.exists(patient_dir):
                apl = APL(reference_structure=reference, other_structure=test)
                apl_pct = apl.get_apl(normalized=True)
                apl_voxel = apl.get_apl(normalized=False)
            else:
                apl_pct = 0.0
                apl_voxel = 0

            obs_metrics[patient][cur_key][Metric.APL_PCT] = apl_pct
            obs_metrics[patient][cur_key][Metric.APL_VOXEL] = apl_voxel

            # surface dice
            if os.path.exists(patient_dir):
                sdsc = SurfaceDice(reference_image=reference, other_image=test)
                obs_metrics[patient][cur_key][Metric.SDSC] = sdsc.get_surface_dice()
            else:
                obs_metrics[patient][cur_key][Metric.SDSC] = 1.0

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
                        obs_metrics[patient][cur_key][metric]
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
    "idl.gtvn_2024.03.11.09.08.22_Jesper_test",
    "idl.gtvt_2024.03.11.09.08.22_Jesper_test",
    "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
    "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
    # "idl.gtvn_2024.03.21.13.07.10_Hanna_test",
    # "idl.gtvt_2024.03.21.13.07.10_Hanna_test",
    # "idl.gtvn_2024.03.22.10.08.33_Kenneth_test",
    # "idl.gtvt_2024.03.22.10.08.33_Kenneth_test",
]
for obs_study_id in obs_study_id_list:
    cal_obs_study_metrics(obs_study_id)
