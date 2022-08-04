import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import global_elems as g
from nested_dict import NestedDict

FIGURE_IDX_FONT_SIZE = 20
DSC_LOW_LIMIT = 0.6


def patients_overview(
    idl_id,
):
    train_result_folder = os.path.join(g.IDL_RESULTS_FOLDER, idl_id)

    score_dict = NestedDict()

    # plt.style.use("bmh")  # put this line before drawing figure

    patient_tumor_size_dict = NestedDict()
    for cur_patient_folder in g.get_sub_folders(
        train_result_folder, key_word="patient="
    ):
        patient_tumor_size_dict[cur_patient_folder] = 0

    # iterate through patients folders
    for cur_patient_folder in g.get_dict_keys(patient_tumor_size_dict):
        # create list of round folders
        round_folder_list = ["baseline"]
        round_folder_list += g.get_sub_folders(
            os.path.join(train_result_folder, cur_patient_folder),
            key_word="round=",
        )
        # initialize score_dict of cur patient
        for score_type in ["dsc", "msd", "hd95"]:
            score_dict[cur_patient_folder][score_type] = []
        # iterate through round folders
        for cur_round_folder in round_folder_list:
            iter_json_list = g.get_sub_files(
                os.path.join(train_result_folder, cur_patient_folder, cur_round_folder),
                key_word=".json",
            )
            best_iter_json = iter_json_list[-1]
            best_score_dict = g.load_json(
                os.path.join(
                    train_result_folder,
                    cur_patient_folder,
                    cur_round_folder,
                    best_iter_json,
                )
            )
            for score_type in ["dsc", "msd", "hd95"]:
                cur_score = best_score_dict[score_type]["3d"]
                score_dict[cur_patient_folder][score_type].append(cur_score)

        # get tumor size of current patient
        label_nii_path = os.path.join(
            train_result_folder, cur_patient_folder, "baseline", "label.nii"
        )
        label_data = g.load_nii(label_nii_path)
        label_data = g.binarize_img(label_data)
        label_size = label_data.sum()
        label_size *= g.NII_SPACING[0] * g.NII_SPACING[1] * g.NII_SPACING[2]
        label_size *= 0.001  # mm3 to cm3
        patient_tumor_size_dict[cur_patient_folder] = label_size

    # sort by tumor size increment
    patient_tumor_size_dict = g.sort_dict_by_value(patient_tumor_size_dict)

    max_label_size = int(max(patient_tumor_size_dict.values()))
    min_label_size = int(min(patient_tumor_size_dict.values()))
    color_map_step = 1
    # color map
    mymap = mpl.colors.LinearSegmentedColormap.from_list("mycolors", ["yellow", "red"])
    # Using contourf to provide my colorbar info, then clearing the figure
    levels = range(min_label_size, max_label_size, color_map_step)
    color_bar = plt.contourf([[0, 0], [0, 0]], levels, cmap=mymap)
    plt.clf()

    # # init colors
    # line_colors = []
    # color_low_limit = 0.2
    # color_gap = (1 - color_low_limit) / (len(patient_tumor_size_dict) - 1)
    # for i in range(len(patient_tumor_size_dict)):
    #     if 1:  # colour deepens as tumor size increases
    #         cur_color = color_gap * (len(patient_tumor_size_dict) - i - 1)
    #     else:  # colour fades as tumor size increases
    #         cur_color = color_gap * i + color_low_limit
    #     line_colors.append((cur_color, cur_color, 1))

    # draw line chart
    for score_type in ["dsc", "msd", "hd95"]:
        print("draw " + score_type + " plt:")
        plt.figure(figsize=(10, 5))
        fig_title = "Performance of iDL ("

        if score_type == "dsc":
            plt.ylim(top=1.0)
            plt.ylabel("DSC")
            fig_title += "Dice similarity coefficient)"

        elif score_type == "msd":
            # plt.ylim(bottom=0)
            plt.ylabel("MSD (mm)")
            fig_title += "Mean surface distance)"

        elif score_type == "hd95":
            # plt.ylim(bottom=0)
            plt.ylabel("HD95 (mm)")
            fig_title += "95% Hausdorff distance)"

        plt.xticks(range(100))
        plt.xlabel("Update Round")
        plt.title(fig_title)

        # plt.yticks(rotation=45)
        # color_idx = 0

        for cur_patient_folder in tqdm(g.get_dict_keys(patient_tumor_size_dict)):
            cur_label_size = patient_tumor_size_dict[cur_patient_folder]
            color_value = (int(cur_label_size) - min_label_size) / (
                max_label_size - min_label_size
            )
            red = 1
            green = 1 - color_value
            blue = 0
            plt.plot(
                range(len(score_dict[cur_patient_folder][score_type])),
                score_dict[cur_patient_folder][score_type],
                "-o",
                color=(red, green, blue),  # line_colors[color_idx],
            )
            # # new color
            # color_idx += 1
            # if color_idx == len(line_colors):
            #     color_idx = 0

        plt.colorbar(color_bar, label="Tumour size (cm³)")

        # plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')

        if score_type == "dsc":
            plt.text(
                x=0,
                y=0.08,
                s="A",
                fontsize=FIGURE_IDX_FONT_SIZE,
                weight="bold",
            )
        elif score_type == "msd":
            plt.text(
                x=2.9,
                y=18.85,
                s="B",
                fontsize=FIGURE_IDX_FONT_SIZE,
                weight="bold",
            )

        img_path = os.path.join(
            os.path.join(
                g.PROJ_PATH,
                "idl_figs",
                "patients.overview." + score_type + ".png",
            )
        )

        # add grid at last because this will need the data
        plt.grid(
            True,
            color="black",
            # linestyle="--",
            # linewidth=0.5,
            # axis="y",
        )

        # set color of axes
        # plt.gca().patch.set_facecolor("mediumaquamarine")
        plt.gca().patch.set_facecolor("darkblue")
        plt.gca().patch.set_alpha(0.65)

        plt.savefig(img_path)


def compare_idl_results(key_hyper: str, idl_id_list: list):
    g.print_line()

    # save avg 3d score of dsc/msd/hd95
    avg_score_dict = NestedDict()

    for cur_idl_id in tqdm(idl_id_list):

        hyper_dict = g.load_json(
            os.path.join(g.IDL_RESULTS_FOLDER, cur_idl_id, "hyper.json")
        )
        avg_score_dict[cur_idl_id] = g.load_json(
            os.path.join(g.IDL_RESULTS_FOLDER, cur_idl_id, "avg_score.json")
        )
        avg_score_dict[cur_idl_id][key_hyper] = hyper_dict[key_hyper]
        select_step = g.str_to_list(hyper_dict["select.step"])

        # after all patients data recorded
        for score_type in ["dsc", "msd", "hd95"]:

            # transfer avg scores from dict to list
            score_list = []

            for cur_round in avg_score_dict[cur_idl_id][score_type]:

                cur_round_score = avg_score_dict[cur_idl_id][score_type][cur_round][
                    "full.slices"
                ]

                # change "round=0X" into "0X"
                cur_round = int(cur_round[len(cur_round) - 2 :])

                # add cur_round_avg_score into cur_type_avg_score_list
                if key_hyper == "select.step":
                    # last round
                    if cur_round >= len(select_step):
                        score_list.append(cur_round_score)

                    # not the last round
                    else:
                        next_round_slices_num = int(select_step[cur_round])
                        for i in range(next_round_slices_num):
                            score_list.append(cur_round_score)
                else:
                    score_list.append(cur_round_score)

            avg_score_dict[cur_idl_id][score_type] = score_list

    for score_type in ["dsc", "msd", "hd95"]:
        plt.figure(figsize=(10, 5))
        fig_title = "Performance of iDL ("

        if score_type == "dsc":
            plt.ylim(DSC_LOW_LIMIT, 1.0)
            plt.ylabel("DSC")
            fig_title += "Dice similarity coefficient)"

        elif score_type == "msd":
            plt.ylim(0.0, 6)
            plt.ylabel("MSD (mm)")
            fig_title += "Mean surface distance)"

        elif score_type == "hd95":
            plt.ylim(0.0, 50)
            plt.ylabel("HD95 (mm)")
            fig_title += "95% Hausdorff distance)"

        if key_hyper == "select.step":
            plt.xlabel("Slices Annotated")
            plt.xticks(range(100))
        else:
            plt.xlabel("Update Round")
            plt.xticks(range(10))

        # set title
        plt.title(fig_title)

        # plt.yticks(rotation=45)
        line_colors = ["c", "b", "g", "y", "r", "m", "k", "w"]
        color_idx = 0
        for cur_idl_id in avg_score_dict:

            # plt_label = key_hyper.replace(".", " ") + " = "

            if key_hyper == "select.step":
                plt_label = "N = "
                plt_label += avg_score_dict[cur_idl_id][key_hyper]

            elif key_hyper == "select.scenario":
                plt_label = "Scenario = "

                if avg_score_dict[cur_idl_id][key_hyper] == "equal.divide":
                    plt_label += "Equal-divide"
                elif avg_score_dict[cur_idl_id][key_hyper] == "largest":
                    plt_label += "Largest"
                elif avg_score_dict[cur_idl_id][key_hyper] == "random":
                    plt_label += "Random"

            elif key_hyper == "loss.hybrid.weight":
                plt_label = "loss function = "
                if avg_score_dict[cur_idl_id][key_hyper] == 1:
                    plt_label += "Dice loss"
                elif avg_score_dict[cur_idl_id][key_hyper] == 0:
                    plt_label += "Focal loss"
                else:
                    plt_label += "Hybrid Focal loss"

            else:
                plt_label = str(avg_score_dict[cur_idl_id][key_hyper])

            plt.plot(
                range(len(avg_score_dict[cur_idl_id][score_type])),  # x axis: round
                avg_score_dict[cur_idl_id][score_type],  # y axis: value
                color=line_colors[color_idx],
                label=plt_label,
            )
            color_idx += 1
            if color_idx == len(line_colors):
                color_idx = 0

        # plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')

        plt.grid()

        if score_type == "dsc":
            plt.text(
                x=0,
                y=DSC_LOW_LIMIT + 0.03,
                s="A",
                fontsize=FIGURE_IDX_FONT_SIZE,
                weight="bold",
            )
            plt.legend(loc="lower right")
        elif score_type == "msd":
            plt.text(
                x=0,
                y=0.4,
                s="B",
                fontsize=FIGURE_IDX_FONT_SIZE,
                weight="bold",
            )
            plt.legend(loc="upper right")
        elif score_type == "hd95":
            plt.legend(loc="upper right")

        img_path = g.create_folder(os.path.join(g.PROJ_PATH, "idl_figs"))
        img_path = os.path.join(
            img_path,
            key_hyper + "." + score_type + ".png",
        )  # ".svg",
        plt.savefig(img_path)
