import os
from pathlib import Path

import global_utils.global_core as g
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Function to plot bar charts and switch AU-CT/MR and NKI bars in the first group
def plot_metrics(csv_path):
    """
    Plot bar charts for given metrics, switching AU-CT/MR and NKI bars only in the first group.

    Parameters:
        csv_path (str): Path to the CSV file containing the dataset.
    """

    # csv_name = os.path.splitext(os.path.basename(csv_path))[0]

    # # Load the data
    # data = pd.read_csv(csv_path)

    # # Extract unique groups and datasets
    # unique_groups = data["Group"].dropna().unique()
    # unique_datasets = data["Dataset"].dropna().unique()

    # # Define distinct colors for datasets
    # distinct_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    # updated_colors = {
    #     dataset: (f"{color}80", color)  # Light for Baseline, dark for iDL
    #     for dataset, color in zip(unique_datasets, distinct_colors)
    # }

    # # Metrics to plot
    # metrics = ["DSC", "MSD(mm)", "HD95(mm)"]

    # # Define the spacing between groups
    # group_spacing = 1.0  # Space between groups

    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # for ax, metric in zip(axes, metrics):
    #     # Filter the data for the current metric
    #     metric_data = data[data["Metric"] == metric]

    #     # Calculate positions for groups with spacing between them
    #     x = np.arange(len(unique_groups)) * group_spacing  # Group positions
    #     bar_width = 0.2  # Width of each bar
    #     total_bars = (
    #         len(unique_datasets) * 2
    #     )  # Each dataset has two bars (Baseline, iDL)
    #     group_width = total_bars * bar_width  # Total width of bars in a group

    #     added_legends = set()  # Track added legends to avoid duplication

    #     for group_idx, group in enumerate(unique_groups):
    #         # Position for the current group
    #         group_position = group_idx * group_spacing

    #         # Define the dataset order for the first group (switch AU-CT/MR and NKI)
    #         if group_idx == 0:
    #             reordered_datasets = [
    #                 (
    #                     dataset
    #                     if dataset not in ["AU-CT/MR", "NKI"]
    #                     else "NKI" if dataset == "AU-CT/MR" else "AU-CT/MR"
    #                 )
    #                 for dataset in unique_datasets
    #             ]
    #         else:
    #             reordered_datasets = unique_datasets

    #         # Plot bars for each dataset in the current group
    #         for i, dataset in enumerate(reordered_datasets):
    #             dataset_data = metric_data[
    #                 (metric_data["Dataset"] == dataset)
    #                 & (metric_data["Group"] == group)
    #             ]

    #             if dataset_data.empty:
    #                 continue  # Skip if no data exists for this dataset and group

    #             # Values for baseline and iDL
    #             baseline_vals = dataset_data["Baseline"].astype(float).values
    #             idl_vals = dataset_data["iDL"].astype(float).values

    #             # Bar positions within the group (adjacent bars for Baseline and iDL)
    #             baseline_positions = (
    #                 group_position - (group_width / 2) + i * 2 * bar_width
    #             )
    #             idl_positions = baseline_positions + bar_width

    #             # Plot bars and add legend if not already added
    #             if (dataset, "Baseline") not in added_legends:
    #                 ax.bar(
    #                     baseline_positions,
    #                     baseline_vals,
    #                     bar_width,
    #                     label=f"{dataset} Baseline",
    #                     color=updated_colors.get(dataset, ("#D3D3D3", "#A9A9A9"))[0],
    #                 )
    #                 added_legends.add((dataset, "Baseline"))
    #             else:
    #                 ax.bar(
    #                     baseline_positions,
    #                     baseline_vals,
    #                     bar_width,
    #                     color=updated_colors.get(dataset, ("#D3D3D3", "#A9A9A9"))[0],
    #                 )

    #             if (dataset, "iDL") not in added_legends:
    #                 ax.bar(
    #                     idl_positions,
    #                     idl_vals,
    #                     bar_width,
    #                     label=f"{dataset} iDL",
    #                     color=updated_colors.get(dataset, ("#D3D3D3", "#A9A9A9"))[1],
    #                 )
    #                 added_legends.add((dataset, "iDL"))
    #             else:
    #                 ax.bar(
    #                     idl_positions,
    #                     idl_vals,
    #                     bar_width,
    #                     color=updated_colors.get(dataset, ("#D3D3D3", "#A9A9A9"))[1],
    #                 )

    #     # Formatting
    #     ax.set_title(metric)
    #     ax.set_ylabel("Value")
    #     ax.legend()

    # # Layout adjustment
    # fig.tight_layout()
    # # Save the plot as PDF and PNG files in the specified directory
    # for file_ext in ["pdf", "png"]:
    #     fig_path = os.path.join(
    #         g.TRAIN_RESULTS_DIR,
    #         f"{csv_name}.{file_ext}",
    #     )
    #     plt.savefig(fig_path, format=file_ext)
