import csv
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
from global_utils.str_lib import (
    DatasetPart,
    DatasetVer,
    Metric,
    ObsStudyGTVnStep,
    ObsStudyGTVtStep,
    Stat,
)
from metric_utils.added_path_len import APL
from metric_utils.metric_func import (
    avg_surface_distance_symmetric,
    dice,
    hausdorff_distance_95,
    surface_dice,
)
from research_utils.research_core import COLOR_LIST, explain_metric
from tqdm import tqdm


def calculate_metrics(obs_study_id: str):
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
        g.replace_char_in_str(
            obs_study_id, 7, "t"
        ),  # obs_study_step.json is under idl.gtvt dir
        "obs_study_step.json",
    )
    obs_study_step = g.load_json(obs_study_step_json_path)

    # select approved patients
    for patient in obs_study_step.keys():
        if (
            gtv == "gtvt"
            and obs_study_step[patient]["gtvt"] == ObsStudyGTVtStep.APPROVED
        ):
            patient_list.append(patient)
        elif (
            gtv == "gtvn"
            and obs_study_step[patient]["gtvn"] == ObsStudyGTVnStep.APPROVED
        ):
            patient_list.append(patient)

    # loop through patients
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
            continue
            # dsc = 1.0
            # msd = 0.0
            # hd95 = 0.0

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


# def create_metrics_table(obs_study_id_list: list):
#     table_path = Dict()
#     table_data = Dict()

#     for i in ["gtvt", "gtvn"]:
#         table_path[i] = os.path.join(
#             g.TRAIN_RESULTS_DIR,
#             "baseline_obs.study",
#             "3d_idl_vs_correct_{}.csv".format(i),
#         )

#         table_data[i] = [
#             ["Metric", "Statistics", "Jesper", "Kenneth", "Hanna"],  # Header row
#         ]
#         for metric in [
#             Metric.DSC,
#             Metric.MSD,
#             Metric.HD95,
#             Metric.APL_PCT,
#             Metric.APL_VOXEL,
#             Metric.SDSC,
#         ]:
#             for stat in [
#                 # Stat.AVG,
#                 Stat.MEDIAN,
#             ]:
#                 cur_item = [metric, stat, "", "", ""]
#                 table_data[i].append(cur_item)

#     for obs_study_id in tqdm(obs_study_id_list):
#         if obs_study_id.startswith("idl.gtvn_"):
#             cur_table_data = table_data["gtvn"]
#         elif obs_study_id.startswith("idl.gtvt_"):
#             cur_table_data = table_data["gtvt"]
#         else:
#             g.error_exit("obs study train id error")

#         obs_study_dir = os.path.join(
#             g.TRAIN_RESULTS_DIR, "baseline_obs.study", obs_study_id
#         )
#         metrics_dict = g.load_json(
#             os.path.join(obs_study_dir, "3d_idl_vs_correct.json")
#         )

#         for metric in [
#             Metric.DSC,
#             Metric.MSD,
#             Metric.HD95,
#             Metric.APL_PCT,
#             Metric.APL_VOXEL,
#             Metric.SDSC,
#         ]:
#             for stat in [
#                 # Stat.AVG,
#                 Stat.MEDIAN,
#             ]:
#                 cur_value = metrics_dict[stat][metric]
#                 for item in cur_table_data:
#                     if item[0] == metric and item[1] == stat:
#                         if "Jesper" in obs_study_id:
#                             item[2] = cur_value
#                         elif "Kenneth" in obs_study_id:
#                             item[3] = cur_value
#                         elif "Hanna" in obs_study_id:
#                             item[4] = cur_value
#                         break

#     for i in ["gtvt", "gtvn"]:
#         with open(table_path[i], "w", newline="") as file:
#             writer = csv.writer(file)
#             writer.writerows(table_data[i])


def plot_metrics(obs_study_id_list: list):
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

    observers_list = ["Jesper", "Kenneth", "Hanna"]
    patients_list = g.load_json(g.DATASET_SPLIT_PATH[DatasetVer.OBS_STUDY])[
        DatasetPart.TEST
    ]
    patients_list = List(patients_list)
    patients_list.remove("462")  # patient 462 is for testing

    # init label of x axis
    x_label = []
    for i in range(1, len(patients_list) + 1):
        x_label.append(str(i))

    if gtv == "gtvn":
        # patient 536 doesnt have gtvn
        del_idx = patients_list.index("536")
        x_label.remove(x_label[del_idx])
        patients_list.remove("536")

    # Set up a 2x3 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 17))

    axes = axes.flatten()

    i = 0
    for metric in tqdm(
        [
            Metric.APL_PCT,
            Metric.SDSC,
            Metric.DSC,
            Metric.MSD,
            Metric.HD95,
        ]
    ):

        fig_data = Dict()
        for observer in observers_list:
            fig_data[observer] = List()

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

            # add patients' metrics
            for patient in patients_list:
                if metrics_dict["patient={}".format(patient)][metric] == {}:
                    fig_data[observer].append(0)
                else:
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
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            ax.bar(
                x=indices + idx * bar_width,  # list
                height=fig_data[observer],  # list
                width=bar_width,
                label="Observer {}".format(observers_list.index(observer) + 1),
                color=color,
            )

            # draw average line
            ax.axhline(
                g.calculate_avg(fig_data[observer]),
                color=color,
                linestyle="--",
                linewidth=2,
            )

        # Configure title and labels
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
        ax.set_xticklabels(x_label)

        # # Add a legend to describe the observers
        # if metric == Metric.APL_PCT:
        #     legend = ax.legend(loc="upper left")
        # elif metric == Metric.SDSC:
        #     legend = ax.legend(loc="lower left")
        # elif metric == Metric.DSC:
        #     legend = ax.legend(loc="lower left")
        # elif metric == Metric.MSD:
        #     legend = ax.legend(loc="upper left")
        # elif metric == Metric.HD95:
        #     legend = ax.legend(loc="upper left")

        # legend.get_frame().set_alpha(1.0)

        # next sub plot
        i += 1

    # turn off axis of the last figure
    # axes[-1].axis("off")
    fig.delaxes(axes[-1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.03),
    )

    # title
    if gtv == "gtvt":
        title_gtv = "GTVt"
    elif gtv == "gtvn":
        title_gtv = "GTVn"
    fig.suptitle(
        """"Initial" vs "Corrected" Segmentation - {}""".format(title_gtv),
        # y=0.95,  # Adjust y for vertical positioning
    )
    # # Adjust top to create more space
    # plt.subplots_adjust(top=0.85)

    # # Adjust layout to prevent overlap and save the entire figure as a PDF
    # plt.tight_layout(rect=[0, 0, 0.00, 0.95])  # Adjust rect to fit the suptitle
    plt.tight_layout()

    # Save the plot as a PDF file in the specified directory
    plt.savefig(fig_path, format="pdf")
