import os

import global_utils.global_core as g
import matplotlib

# Prevent matplotlib.pyplot from using a GUI (like X11) for rendering.
# Without this line, using breakpoints under X11 without VCXSRV can cause the debugger to freeze.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from global_utils.custom_dict import Dict
from global_utils.custom_list import List
from global_utils.str_lib import DatasetVer, Metric, Stat
from metric_utils.metric_func import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
)

JESPER_GTVT_ID = "idl.gtvt_2024.10.07.13.23.47_Jesper"
KENNETH_GTVT_ID = "idl.gtvt_2024.10.11.11.09.32_Kenneth"
HANNA_GTVT_ID = "idl.gtvt_2024.10.17.10.22.39_Hanna"
JESPER_GTVN_ID = "idl.gtvn_2024.10.07.13.23.47_Jesper"
KENNETH_GTVN_ID = "idl.gtvn_2024.10.11.11.09.32_Kenneth"
HANNA_GTVN_ID = "idl.gtvn_2024.10.17.10.22.39_Hanna"


COLOR_LIST = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#FFD700",  # Gold
    "#808080",  # Gray
    "#00BFFF",  # Deep Sky Blue
    "#6A5ACD",  # Light Slate Blue
]


FONT_SIZE = 18
TITLE_SIZE = 25


def update_font_size():
    plt.rcParams.update(
        {
            "axes.titlesize": FONT_SIZE,  # Title font size
            "axes.labelsize": FONT_SIZE,  # X and Y label font size
            "xtick.labelsize": FONT_SIZE,  # X tick label font size
            "ytick.labelsize": FONT_SIZE,  # Y tick label font size
            "legend.fontsize": FONT_SIZE,  # Legend font size
            "figure.titlesize": TITLE_SIZE,  # Figure title font size
        }
    )


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


def calculate_idl_gtvs_metric(idl_gtvt_id: str, idl_gtvn_id: str):
    baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, "baseline_au")
    idl_gtvt_dir = os.path.join(baseline_dir, idl_gtvt_id)
    idl_gtvn_dir = os.path.join(baseline_dir, idl_gtvn_id)

    metric_dict = Dict()
    for metric_type in [Metric.DSC, Metric.MSD, Metric.HD95]:
        metric_dict[Stat.AVG][metric_type] = []
        metric_dict[Stat.MEDIAN][metric_type] = []

    patient_list = List(g.load_json(g.DATASET_SPLIT_PATH[DatasetVer.AU_EXT])["test"])
    for patient in patient_list:
        print(patient)

        # load idl pred
        gtvt_pred = g.binarize_img(
            g.load_nii(
                os.path.join(
                    idl_gtvt_dir,
                    "patients",
                    "patient={}".format(patient),
                    "round=01",
                    "gtvt_pred.nii.gz",
                )
            )
        )
        print(
            "gtvt_pred",
            gtvt_pred.min(),
            gtvt_pred.max(),
            gtvt_pred.sum(),
            gtvt_pred.shape,
        )

        gtvn_pred = g.binarize_img(
            g.load_nii(
                os.path.join(
                    idl_gtvn_dir,
                    "patients",
                    "patient={}".format(patient),
                    "round=01",
                    "gtvn_pred.nii.gz",
                )
            )
        )
        print(
            "gtvn_pred",
            gtvn_pred.min(),
            gtvn_pred.max(),
            gtvn_pred.sum(),
            gtvn_pred.shape,
        )

        gtvs_pred = np.maximum(gtvt_pred, gtvn_pred)
        print(
            "gtvs_pred",
            gtvs_pred.min(),
            gtvs_pred.max(),
            gtvs_pred.sum(),
            gtvs_pred.shape,
        )

        # load label
        gtvs_label = g.load_gtv_labels(
            dataset_ver=DatasetVer.AU_EXT,
            patient=patient,
        )["gtvs"]
        print(
            "gtvs_label",
            gtvs_label.min(),
            gtvs_label.max(),
            gtvs_label.sum(),
            gtvs_label.shape,
        )

        for metric_type in [Metric.DSC, Metric.MSD, Metric.HD95]:
            if metric_type == Metric.DSC:
                metric_num = dice(
                    test=gtvs_pred,
                    reference=gtvs_label,
                    nan_for_nonexisting=False,
                )
            elif metric_type == Metric.MSD:
                metric_num = avg_surface_distance_symmetric(
                    test=gtvs_pred,
                    reference=gtvs_label,
                    none_for_nonexisting=True,
                    voxel_spacing=g.NII_SPACING,
                )
            elif metric_type == Metric.HD95:
                metric_num = hausdorff_distance_95(
                    test=gtvs_pred,
                    reference=gtvs_label,
                    none_for_nonexisting=True,
                    voxel_spacing=g.NII_SPACING,
                )
            metric_dict["patient={}".format(patient)][metric_type] = metric_num
            metric_dict[Stat.AVG][metric_type].append(metric_num)
            metric_dict[Stat.MEDIAN][metric_type].append(metric_num)

    for metric_type in [Metric.DSC, Metric.MSD, Metric.HD95]:
        avg = g.calculate_avg(metric_dict[Stat.AVG][metric_type])
        metric_dict[Stat.AVG][metric_type] = avg
        median = g.calculate_median(metric_dict[Stat.MEDIAN][metric_type])
        metric_dict[Stat.MEDIAN][metric_type] = median

    g.save_json(metric_dict, os.path.join(baseline_dir, "inference_au_gtvs.json"))


update_font_size()
