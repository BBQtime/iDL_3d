import os
from pathlib import Path

import global_utils.global_core as g
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from global_utils.custom_dict import Dict
from global_utils.str_lib import Metric
from research_utils.research_core import COLOR_LIST, explain_metric


def plot_boxplots(gtv: str):
    if gtv not in ["gtvt", "gtvn"]:
        g.error_exit("Invalid gtv value!")

    origin_data = Dict()

    if gtv == "gtvt":
        # baseline group
        au_baseline_4m = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            "baseline",
            "inference_au.ext_test.json",
        )
        au_baseline_3m = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au_no.pt",
            "baseline",
            "inference_au.ext_test.json",
        )
        nki_baseline = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            "baseline",
            "inference_nki_test.json",
        )
        mda_baseline = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au_no.pt",
            "baseline",
            "inference_mda_test.json",
        )

        # idl group
        au_idl_4m = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            "idl.gtvt_au.ext",
            "inference_au.ext_test.json",
        )
        au_idl_3m = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au_no.pt",
            "idl.gtvt_au.ext_no.pt",
            "inference_au.ext_test.json",
        )
        nki_idl = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            "idl.gtvt_nki",
            "inference_nki_test.json",
        )
        mda_idl = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au_no.pt",
            "idl.gtvt_mda_no.pt",
            "inference_mda_test.json",
        )

        # from scratch group
        nki_scratch = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_nki.new",
            "idl.gtvt_nki.new",
            "inference_nki_test.json",
        )
        mda_scratch = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_mda.new",
            "idl.gtvt_mda.new",
            "inference_mda_test.json",
        )

        # transfer learning group
        nki_transfer = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_nki.transfer",
            "idl.gtvt_nki.transfer",
            "inference_nki_test.json",
        )
        mda_transfer = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_mda.transfer",
            "idl.gtvt_mda.transfer",
            "inference_mda_test.json",
        )

    elif gtv == "gtvn":
        # idl group
        au_idl_4m = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            "idl.gtvn_au_multi.clicks",
            "inference_au.ext_test.json",
        )
        au_idl_3m = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au_no.pt",
            "idl.gtvn_au_no.pt_multi.clicks",
            "inference_au.ext_test.json",
        )
        nki_idl = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au",
            "idl.gtvn_au_multi.clicks",
            "inference_nki_test.json",
        )
        mda_idl = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_au_no.pt",
            "idl.gtvn_au_no.pt_multi.clicks",
            "inference_mda_test.json",
        )

        # from scratch group
        nki_scratch = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_nki.new",
            "idl.gtvn_nki.new_multi.clicks",
            "inference_nki_test.json",
        )
        mda_scratch = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_mda.new",
            "idl.gtvn_mda.new_multi.clicks",
            "inference_mda_test.json",
        )

        # transfer learning group
        nki_transfer = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_nki.transfer",
            "idl.gtvn_nki.transfer_multi.clicks",
            "inference_nki_test.json",
        )
        mda_transfer = os.path.join(
            g.TRAIN_RESULTS_DIR,
            "baseline_mda.transfer",
            "idl.gtvn_mda.transfer_multi.clicks",
            "inference_mda_test.json",
        )

    result_id_list = [
        au_idl_4m,
        au_idl_3m,
        nki_idl,
        mda_idl,
        nki_scratch,
        mda_scratch,
        nki_transfer,
        mda_transfer,
    ]

    if gtv == "gtvt":
        result_id_list = [
            au_baseline_4m,
            au_baseline_3m,
            nki_baseline,
            mda_baseline,
        ] + result_id_list

    categories = ["iDL", "Scratch", "Transfer"]
    if gtv == "gtvt":
        categories = ["Baseline"] + categories

    for result_id in result_id_list:
        origin_data[result_id] = g.load_json(result_id)

    labels = ["AU PET/CT/MR", "AU CT/MR", "NKI", "MDA"]

    x = np.array([0, 1.5, 2.5, 3.5]) if gtv == "gtvt" else np.array([1, 2, 3])
    bar_width = 0.3 if gtv == "gtvt" else 0.25

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(f"{gtv[:-1].upper() + gtv[-1]} Metrics across datasets")

    metric_type_list = [Metric.DSC, Metric.MSD, Metric.HD95]

    for ax, metric_type in zip(axes, metric_type_list):

        plot_data = Dict()

        # baseline ids
        if gtv == "gtvt":
            for baseline_id in [
                au_baseline_4m,
                au_baseline_3m,
                nki_baseline,
                mda_baseline,
            ]:
                plot_data[baseline_id] = []
                for patient in origin_data[baseline_id]:
                    if "patient=" not in patient:
                        continue
                    cur_data = origin_data[baseline_id][patient][gtv][metric_type]
                    if g.is_number(cur_data):
                        plot_data[baseline_id].append(cur_data)
                    # else:
                    #     print("not number")

        # idl ids
        for idl_id in [
            au_idl_4m,
            au_idl_3m,
            nki_idl,
            mda_idl,
            nki_scratch,
            mda_scratch,
            nki_transfer,
            mda_transfer,
        ]:
            plot_data[idl_id] = []
            for patient in origin_data[idl_id]:
                if "patient=" not in patient:
                    continue
                cur_data = origin_data[idl_id][patient][metric_type]["round=01"]
                if g.is_number(cur_data):
                    plot_data[idl_id].append(cur_data)
                # else:
                #     print("not number")

        grouped_data = [
            # idl group
            [
                plot_data[au_idl_4m],
                plot_data[au_idl_3m],
                plot_data[nki_idl],
                plot_data[mda_idl],
            ],
            # scratch group
            [[], [], plot_data[nki_scratch], plot_data[mda_scratch]],
            # transfer group
            [[], [], plot_data[nki_transfer], plot_data[mda_transfer]],
        ]
        if gtv == "gtvt":
            # add baseline group
            grouped_data = [
                [
                    plot_data[au_baseline_4m],
                    plot_data[au_baseline_3m],
                    plot_data[nki_baseline],
                    plot_data[mda_baseline],
                ]
            ] + grouped_data

        for i, label in enumerate(labels):
            data_for_plot = [group[i] for group in grouped_data]
            pos = x + (i - 1.5) * bar_width
            valid_data = [d for d in data_for_plot if d]
            valid_pos = [p for d, p in zip(data_for_plot, pos) if d]

            ax.boxplot(
                valid_data,
                positions=valid_pos,
                widths=bar_width,
                patch_artist=True,
                boxprops=dict(facecolor=COLOR_LIST[i], color=COLOR_LIST[i]),
                whiskerprops=dict(color=COLOR_LIST[i]),
                capprops=dict(color=COLOR_LIST[i]),
                medianprops=dict(color="white", linewidth=2),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20)
        ax.set_title(explain_metric(metric_type))

    # Create legend handles manually
    legend_handles = [
        mpatches.Patch(color=COLOR_LIST[i], label=label)
        for i, label in enumerate(labels)
    ]
    fig.legend(
        legend_handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
    )

    plt.tight_layout()

    # Adjust top to create more space
    # Adjust spacing between rows
    # (after tight_layout())
    plt.subplots_adjust(top=0.72, hspace=0.5)

    # Save the plot as PDF and PNG files in the specified directory
    for file_ext in ["pdf", "png"]:
        fig_path = os.path.join(
            g.TRAIN_RESULTS_DIR,
            f"cross_dataset_metrics_{gtv}.{file_ext}",
        )
        plt.savefig(fig_path, format=file_ext)
