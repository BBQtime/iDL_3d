import csv
import os

from custom import Explorer
from custom import Global as g
from custom import Json, List


def compare_idls_in_table(
    baseline_id: str, idl_results_list: list, table_name: str = "idl_compare"
):
    # field names
    fields = ["Patient"]
    for metric in ["DSC", "MSD", "HD95"]:
        for i in range(len(idl_results_list)):
            fields.append("{}_{}".format(metric, i + 1))

    baseline_dir = os.path.join(g.TRAIN_RESULTS_DIR, baseline_id)
    fold_dir = Explorer.get_sub_folders(baseline_dir, "fold=", full_path=True)[0]
    epoch_dir = Explorer.get_sub_folders(fold_dir, "epoch=", full_path=True)[0]
    idl_gtvt_main_dir = os.path.join(epoch_dir, "idl_gtvt")

    patient_list = Json.load(
        os.path.join(
            idl_gtvt_main_dir, idl_results_list[0], "inference_test_inter.json"
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
                os.path.join(idl_gtvt_main_dir, idl_id, "inference_test_inter.json")
            )
            for metric in ["DSC", "MSD", "HD95"]:
                cur_patient_score["{}_{}".format(metric, i + 1)] = gtvt_score[patient][
                    metric.lower()
                ]["round=01"]

        score.append(cur_patient_score)

    table_path = os.path.join(idl_gtvt_main_dir, table_name + ".csv")

    with open(table_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score)


compare_idls_in_table(
    baseline_id="baseline_2023.02.27.07.08.09_loss.delta=0.5_loss.gamma=0.5_optimal",
    idl_results_list=[
        "idl_gtvt_2023.04.04.14.43.54_fp.fn:4.0_no.post.processing",
        "idl_gtvt_2023.04.04.14.43.54_fp.fn:4.0",
    ],
)
