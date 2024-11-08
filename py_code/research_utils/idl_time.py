import os

import global_utils.global_core as g
import matplotlib
from global_utils.custom_list import List
from global_utils.str_lib import DatasetPart, DatasetVer
from ui_utils.obs_study_timer import ObsStudyTimer

# Prevent matplotlib.pyplot from using a GUI (like X11) for rendering.
# Without this line, using breakpoints under X11 without VCXSRV can cause the debugger to freeze.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from global_utils.custom_dict import Dict
from research_utils.research_core import COLOR_LIST
from tqdm import tqdm


def __seconds_to_minutes_decimal(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    decimal_minutes = remaining_seconds / 60
    total_minutes = minutes + decimal_minutes
    return total_minutes


def __time_str_to_seconds(time_str: str):
    h, m, s = map(int, time_str.split(":"))
    seconds = h * 3600 + m * 60 + s
    return seconds


def __explain_idl_step(idl_step: str):
    if idl_step == ObsStudyTimer.CLICK_GTVN_CENTERS:
        return "Click GTVn Centers"
    elif idl_step == ObsStudyTimer.CLICK_GTVT_CENTER:
        return "Click GTVt Center"
    elif idl_step == ObsStudyTimer.CORRECT_GTVT:
        return "Correct GTVt"
    elif idl_step == ObsStudyTimer.CORRECT_GTVN:
        return "Correct GTVn"
    elif idl_step == ObsStudyTimer.DELINEATE_GTVT:
        return "Delineate GTVt"
    elif idl_step == ObsStudyTimer.WAIT_GTVN_PRED:
        return "AI Generate GTVn"
    elif idl_step == ObsStudyTimer.WAIT_GTVT_PRED:
        return "AI Generate GTVt"
    else:
        return None


def plot_time_per_patient(obs_study_gtvt_id_list: list):
    patients_list = g.load_json(g.DATASET_SPLIT_PATH[DatasetVer.OBS_STUDY])[
        DatasetPart.TEST
    ]
    patients_list = List(patients_list)
    patients_list.remove("462")  # patient 462 is for testing

    observers_list = ["Jesper", "Kenneth", "Hanna"]

    fig_data = Dict()
    for observer in observers_list:
        fig_data[observer] = []

    # loop through observer study train id
    for obs_study_gtvt_id in tqdm(obs_study_gtvt_id_list):
        if not obs_study_gtvt_id.startswith("idl.gtvt_"):
            g.error_exit("Must be an 'idl.gtvt' id!")
        json_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            obs_study_gtvt_id,
            "time_used.json",
        )
        time_dict = g.load_json(json_path)

        # get observer name from train id
        for observer in observers_list:
            if observer in obs_study_gtvt_id:
                break

        for patient in patients_list:
            total_time = time_dict["patient={}".format(patient)]["total.time"]
            total_time = __time_str_to_seconds(total_time)
            total_time = __seconds_to_minutes_decimal(total_time)
            fig_data[observer].append(total_time)

        # # calculate avg and mean
        # avg = g.calculate_avg(fig_data[observer])
        # avg = round(avg)
        # median = g.calculate_median(fig_data[observer])
        # median = round(median)
        # fig_data[observer].append(avg)
        # fig_data[observer].append(median)

        # for i in range(len(fig_data[observer])):
        #     fig_data[observer][i] = __seconds_to_minutes_decimal(fig_data[observer][i])

    # Set up a 2x3 grid of subplots
    _, ax = plt.subplots(figsize=(12, 8))

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
    ax.set_xlabel("Patients")
    ax.set_ylabel("Minutes")
    ax.set_title("Median Time Used per Patient")

    # Set x-axis ticks to be centered under each group of bars
    ax.set_xticks(indices + bar_width)
    # init label of x axis
    x_label = []
    for i in range(1, len(patients_list) + 1):
        x_label.append(str(i))
    ax.set_xticklabels(x_label)

    # Add a legend to describe the observers
    # ax.legend()
    legend = ax.legend(loc="best")  # "upper right")
    legend.get_frame().set_alpha(0.3)

    # Adjust layout to prevent overlap and save the entire figure as a PDF
    plt.tight_layout()

    # Save the plot as a PDF file in the specified directory
    fig_path = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", "time_per_patient.pdf"
    )
    plt.savefig(fig_path, format="pdf")


def plot_time_per_step(obs_study_gtvt_id_list: list):
    observers_list = ["Jesper", "Kenneth", "Hanna"]
    idl_step_list = [
        ObsStudyTimer.CLICK_GTVT_CENTER,
        ObsStudyTimer.DELINEATE_GTVT,
        ObsStudyTimer.WAIT_GTVT_PRED,
        ObsStudyTimer.CLICK_GTVN_CENTERS,
        ObsStudyTimer.WAIT_GTVN_PRED,
        ObsStudyTimer.CORRECT_GTVT,
        ObsStudyTimer.CORRECT_GTVN,
        # ObsStudyTimer.PATIENT_TOTAL_TIME,
    ]

    # Set up a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(
        "Mean Time Usage per iDL Step",
    )
    axes = axes.flatten()

    sub_fig_idx = 0
    # loop through observer study train id
    for obs_study_gtvt_id in tqdm(obs_study_gtvt_id_list):
        if not obs_study_gtvt_id.startswith("idl.gtvt_"):
            g.error_exit("Must be an 'idl.gtvt' id!")

        # init data
        fig_data = Dict()
        for idl_step in idl_step_list:
            fig_data[idl_step]["value"] = []

        # load json
        json_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_obs.study",
            obs_study_gtvt_id,
            "time_used.json",
        )
        time_dict = g.load_json(json_path)

        # calculate avrage time used of each step
        for patient in time_dict.keys():
            for idl_step in idl_step_list:
                seconds = __time_str_to_seconds(time_dict[patient][idl_step])
                fig_data[idl_step]["value"].append(seconds)

        for idl_step in idl_step_list:
            fig_data[idl_step]["value"] = g.calculate_avg(fig_data[idl_step]["value"])

        # set start points of each step
        fig_data[ObsStudyTimer.CLICK_GTVT_CENTER]["start"] = 0
        fig_data[ObsStudyTimer.DELINEATE_GTVT]["start"] = fig_data[
            ObsStudyTimer.CLICK_GTVT_CENTER
        ]["value"]
        fig_data[ObsStudyTimer.WAIT_GTVT_PRED]["start"] = fig_data[
            ObsStudyTimer.CLICK_GTVN_CENTERS
        ]["start"] = (
            fig_data[ObsStudyTimer.DELINEATE_GTVT]["start"]
            + fig_data[ObsStudyTimer.DELINEATE_GTVT]["value"]
        )
        fig_data[ObsStudyTimer.WAIT_GTVN_PRED]["start"] = (
            fig_data[ObsStudyTimer.CLICK_GTVN_CENTERS]["start"]
            + fig_data[ObsStudyTimer.CLICK_GTVN_CENTERS]["value"]
        )

        # check wether correct.gtvt or correct.gtvn started first
        wait_gtvt_end_point = (
            fig_data[ObsStudyTimer.WAIT_GTVT_PRED]["start"]
            + fig_data[ObsStudyTimer.WAIT_GTVT_PRED]["value"]
        )
        wait_gtvn_end_point = (
            fig_data[ObsStudyTimer.WAIT_GTVN_PRED]["start"]
            + fig_data[ObsStudyTimer.WAIT_GTVN_PRED]["value"]
        )
        # gtvt pred comes first
        if wait_gtvt_end_point < wait_gtvn_end_point:
            fig_data[ObsStudyTimer.CORRECT_GTVT]["start"] = wait_gtvt_end_point
            fig_data[ObsStudyTimer.CORRECT_GTVN]["start"] = max(
                wait_gtvn_end_point,
                wait_gtvt_end_point + fig_data[ObsStudyTimer.CORRECT_GTVT]["value"],
            )
        # gtvn pred comes first
        else:
            fig_data[ObsStudyTimer.CORRECT_GTVN]["start"] = wait_gtvn_end_point
            fig_data[ObsStudyTimer.CORRECT_GTVT]["start"] = max(
                wait_gtvt_end_point,
                wait_gtvn_end_point + fig_data[ObsStudyTimer.CORRECT_GTVN]["value"],
            )

        # seconds to minutes
        for idl_step in idl_step_list:
            for i in ["start", "value"]:
                fig_data[idl_step][i] = __seconds_to_minutes_decimal(
                    fig_data[idl_step][i]
                )

        # create sub fig
        ax = axes[sub_fig_idx]
        bar_height = 5
        step_space = 2  # Space between bars
        total_height = len(idl_step) * (bar_height + step_space)
        y_positions = [
            total_height - (i * (bar_height + step_space))
            for i in range(len(idl_step_list))
        ]

        # set title
        for observer in observers_list:
            if observer in obs_study_gtvt_id:
                break
        ax.set_title(
            "Observer {}".format(observers_list.index(observer) + 1),
        )
        ax.set_xlabel("Minutes")
        ax.set_yticks([y + bar_height / 2 for y in y_positions])
        y_labels = []
        for idl_step in idl_step_list:
            y_labels.append(__explain_idl_step(idl_step))
        ax.set_yticklabels(y_labels, rotation=30)
        ax.grid(True)

        # set limit of x axis
        ax.set_xlim(-0.5, 20)

        idx = 0
        for idl_step in idl_step_list:
            lower = fig_data[idl_step]["start"]
            value = fig_data[idl_step]["value"]
            ax.broken_barh(
                [(lower, value)],
                (y_positions[idx], bar_height),
                facecolors=COLOR_LIST[idx],
            )
            idx += 1

        # next sub plot
        sub_fig_idx += 1

    # turn off axis of the last figure
    # axes[-1].axis("off")
    fig.delaxes(axes[-1])

    # Adjust layout to prevent overlap and save the entire figure as a PDF
    plt.tight_layout()
    # Save the plot as a PDF file in the specified directory
    fig_path = os.path.join(
        g.TRAIN_RESULTS_DIR, "baseline_obs.study", "time_per_step.pdf"
    )
    plt.savefig(fig_path, format="pdf")
