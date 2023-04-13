import csv
import os
from custom import Global as g
from custom import Json
from custom import List


def compare_idls_in_table(
    baseline_id: str, idl_results_list: list, table_name: str = "idl_compare"
):

    # field names
    fields = ["Patient"]
    for metric in ["DSC", "MSD", "HD95"]:
        for i in range(len(idl_results_list)):
            fields.append("{}_{}".format(metric, i + 1))

    patient_list = Json.load(
        os.path.join(
            g.TRAIN_RESULTS_FOLDER, baseline_id, idl_results_list[0], "score_gtvt.json"
        )
    )
    patient_list = patient_list.keys()

    score = List()

    for patient in patient_list:
        cur_patient_score = {}
        cur_patient_score["Patient"] = patient

        for i in range(len(idl_results_list)):
            idl_id = idl_results_list[i]
            gtvt_score = Json.load(
                os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, idl_id, "score.json")
            )
            for metric in ["DSC", "MSD", "HD95"]:
                cur_patient_score["{}_{}".format(metric, i + 1)] = gtvt_score[patient][
                    metric.lower()
                ]["round=01"]

        score.append(cur_patient_score)

    table_path = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, table_name + ".csv")

    with open(table_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score)


compare_idls_in_table(
    baseline_id="baseline_2023.02.27.07.08.09_loss.delta=0.5_loss.gamma=0.5_optimal",
    idl_results_list=[
        "idl_2023.03.14.23.28.32_select.scenario={'coronal':'equal.divide','sagittal':'equal.divide','transverse':'equal.divide'}",
        "idl_2023.03.15.02.01.26_select.scenario={'coronal':'gravity.center','sagittal':'gravity.center','transverse':'gravity.center'}",
    ],
)
