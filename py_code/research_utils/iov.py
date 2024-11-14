import os

import global_utils.global_core as g
import matplotlib

# Prevent matplotlib.pyplot from using a GUI (like X11) for rendering.
# Without this line, using breakpoints under X11 without VCXSRV can cause the debugger to freeze.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetPart, DatasetVer, Metric, Stat
from metric_utils.added_path_len import APL
from metric_utils.metric_func import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
    surface_dice,
)
from research_utils.research_core import (
    FONT_SIZE,
    explain_metric,
    get_obs_study_patients_list,
)
from tqdm import tqdm


def calculate_iov(obs_study_id_1: str, obs_study_id_2: str):
    print(f"{obs_study_id_1} vs {obs_study_id_2}")
    obs_study_id = Dict()
    obs_study_id["1"] = obs_study_id_1
    obs_study_id["2"] = obs_study_id_2

    if obs_study_id["1"] == obs_study_id["2"]:
        g.error_exit("2 input obs_study_ids cannot be identical.")

    if obs_study_id["1"].startswith("idl.gtvn_") and obs_study_id["2"] == "label":
        obs_study_id["2"] = obs_study_id["2"].capitalize()
        gtv = "gtvn"
    elif obs_study_id["1"].startswith("idl.gtvt_") and obs_study_id["2"] == "label":
        obs_study_id["2"] = obs_study_id["2"].capitalize()
        gtv = "gtvt"
    elif obs_study_id["1"] == "label" and obs_study_id["2"].startswith("idl.gtvn_"):
        obs_study_id["1"] = obs_study_id["1"].capitalize()
        gtv = "gtvn"
    elif obs_study_id["1"] == "label" and obs_study_id["2"].startswith("idl.gtvt_"):
        obs_study_id["1"] = obs_study_id["1"].capitalize()
        gtv = "gtvt"

    elif obs_study_id["1"].startswith("idl.gtvn_") and obs_study_id["2"].startswith(
        "idl.gtvn_"
    ):
        gtv = "gtvn"
    elif obs_study_id["1"].startswith("idl.gtvt_") and obs_study_id["2"].startswith(
        "idl.gtvt_"
    ):
        gtv = "gtvt"
    else:
        g.error_exit("obs study train id error")

    # init observers
    observer = Dict()
    # get observer from obs study id
    for name in ["Jesper", "Kenneth", "Hanna", "Label"]:
        for idx in ["1", "2"]:
            if name in obs_study_id[idx]:
                observer[idx] = name
    if observer["1"] == observer["2"]:
        g.error_exit("2 observers cannot be identical.")

    # init patients
    patients_list = get_obs_study_patients_list()
    if gtv == "gtvn":
        # patient 536 doesnt have gtvn
        patients_list.remove("536")

    # init metrics dict
    metrics_dict = Dict()

    for result_type in ["idl", "correct"]:
        for stat in [Stat.AVG, Stat.MEDIAN]:
            for metric in [
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
                Metric.APL_PCT,
                Metric.APL_VOXEL,
                Metric.SDSC,
            ]:
                metrics_dict[result_type][stat][metric] = []

        for patient in tqdm(patients_list):
            patient = "patient={}".format(patient)

            img_data = Dict()
            for idx in ["1", "2"]:
                if observer[idx] == "Label":
                    img_data[idx] = g.load_nii(
                        os.path.join(
                            g.DATASET_DIR[DatasetVer.OBS_STUDY],
                            "HNCDL_{}_GTV{}.nii".format(
                                patient[len("patient=") :], gtv[-1]
                            ),
                        ),
                        binary=True,
                    )
                else:
                    idl_img_dir = os.path.join(
                        g.TRAIN_RESULTS_DIR,
                        "baseline_obs.study",
                        obs_study_id[idx],
                        "patients",
                        patient,
                        "round=01",
                    )
                    if os.path.exists(idl_img_dir):
                        origin_pred = g.load_nii(
                            os.path.join(idl_img_dir, "{}_pred.nii.gz".format(gtv)),
                            binary=True,
                        )
                        if result_type == "idl":
                            img_data[idx] = origin_pred
                        elif result_type == "correct":
                            correction_mask = g.load_nii(
                                os.path.join(
                                    idl_img_dir,
                                    "{}_correction_mask.nii.gz".format(gtv),
                                ),
                                binary=True,
                            )
                            correction = g.load_nii(
                                os.path.join(
                                    idl_img_dir, "{}_correction.nii.gz".format(gtv)
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

            # both imgs are not empty
            if img_data["1"] is not None and img_data["2"] is not None:
                if img_data["1"].shape != img_data["2"].shape:
                    g.error_exit("Img size of observer 1 and 2 are different!")
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
            # both imgs are empty
            elif img_data["1"] is None and img_data["2"] is None:
                dsc = None
                msd = None
                hd95 = None
                apl_pct = None
                apl_voxel = None
                sdsc = None
            # one of imgs is empty
            else:
                dsc = 0.0
                msd = None
                hd95 = None
                apl_pct = 1.0
                apl_voxel = None
                sdsc = 0.0

            metrics_dict[result_type][patient][Metric.DSC] = dsc
            metrics_dict[result_type][patient][Metric.MSD] = msd
            metrics_dict[result_type][patient][Metric.HD95] = hd95
            metrics_dict[result_type][patient][Metric.APL_PCT] = apl_pct
            metrics_dict[result_type][patient][Metric.APL_VOXEL] = apl_voxel
            metrics_dict[result_type][patient][Metric.SDSC] = sdsc

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
                    metrics_dict[result_type][stat][metric].append(
                        metrics_dict[result_type][patient][metric]
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
            metrics_dict[result_type][Stat.MEDIAN][metric] = g.calculate_median(
                metrics_dict[result_type][Stat.MEDIAN][metric]
            )
            metrics_dict[result_type][Stat.AVG][metric] = g.calculate_avg(
                metrics_dict[result_type][Stat.AVG][metric]
            )

    # save json
    metrics_path = os.path.join(
        g.TRAIN_RESULTS_DIR,
        "baseline_obs.study",
        "iov_{}_vs_{}.json".format(observer["1"], observer["2"]),
    )
    if os.path.exists(metrics_path):
        final_dict = g.load_json(metrics_path)
    else:
        final_dict = Dict()
    final_dict[gtv] = metrics_dict
    g.save_json(data=final_dict, path=metrics_path)


def plot_iov():
    observer_list = List(["Jesper", "Kenneth", "Hanna", "Label"])
    label_symbol = ["1", "2", "3", "Label"]
    label_text = ["Observer 1", "Observer 2", "Observer 3", "Clinical Label"]
    legend_text = "\n".join(
        [f"{label_symbol[i]} = {label_text[i]}" for i in range(len(label_text))]
    )

    obs_study_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_obs.study")
    for gtv in ["gtvt", "gtvn"]:
        if gtv == "gtvt":
            title_gtv = "GTVt"
        elif gtv == "gtvn":
            title_gtv = "GTVn"

        for result_type in [
            # "idl",
            "correct",
        ]:
            if result_type == "idl":
                title_img_name = """"Initial" Segmentations"""
            elif result_type == "correct":
                title_img_name = """"Corrected" Segmentations"""

            # Setting up the figure and axes for a 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(20, 13))
            fig.suptitle(
                "IOV between {} - {}".format(title_img_name, title_gtv),
            )
            # Flattening the axes array for easier iteration
            axes = axes.flatten()

            i = 0
            for metric in [
                Metric.APL_PCT,
                Metric.SDSC,
                Metric.DSC,
                Metric.MSD,
                Metric.HD95,
            ]:
                if metric in [Metric.SDSC, Metric.DSC]:
                    iov_matrix = np.ones(shape=(4, 4))
                else:
                    iov_matrix = np.zeros(shape=(4, 4))

                for observer_1, observer_2 in observer_list.get_combinations():
                    json_path = os.path.join(
                        obs_study_dir,
                        "iov_{}_vs_{}.json".format(observer_1, observer_2),
                    )
                    data = g.load_json(json_path)
                    data = data[gtv][result_type]["median"][metric]

                    x = observer_list.index(observer_1)
                    y = observer_list.index(observer_2)
                    iov_matrix[x][y] = None
                    iov_matrix[y][x] = data

                # Creating the heatmap with a white-to-blue color gradient
                if metric in [Metric.SDSC, Metric.DSC, Metric.APL_PCT]:
                    vmax = 1.0
                    vmin = 0.0
                # elif metric ==:
                #     vmax=0.0
                #     vmin=0.0
                else:
                    vmax = vmin = None

                sns.heatmap(
                    iov_matrix,
                    ax=axes[i],
                    annot=True,
                    cmap="Blues",
                    square=True,
                    cbar=True,
                    vmin=vmin,
                    vmax=vmax,
                    annot_kws={"size": FONT_SIZE},
                )

                subtitle = explain_metric(metric)
                axes[i].set_title(subtitle)
                axes[i].set_xticklabels(
                    label_symbol,
                )
                axes[i].set_yticklabels(
                    label_symbol,
                    # rotation=0,
                )

                i += 1

            # turn off axis of the last figure
            # axes[-1].axis("off")
            fig.delaxes(axes[-1])

            # add legend
            plt.figtext(
                0.96,
                0.05,
                legend_text,
                ha="right",
                va="bottom",
                fontsize=FONT_SIZE,
                bbox={
                    "facecolor": "white",
                    "alpha": 1.0,
                    "pad": 5,
                    "edgecolor": "gray",
                },
            )

            plt.tight_layout()

            for file_ext in ["pdf", "png"]:
                fig_path = os.path.join(
                    obs_study_dir, f"iov_{gtv}_{result_type}.{file_ext}"
                )
                plt.savefig(fig_path, format=file_ext)
