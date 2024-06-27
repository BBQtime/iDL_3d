import global_core as g
from custom_list import List
from research_analysis import (
    calculate_3d_idl_vs_correct,
    calculate_gtvt_input_variation,
    calculate_gtvt_slices_metrics,
    calculate_idl_gtvs_metric,
    calculate_iov,
    create_table_3d_idl_vs_correct,
    create_table_gtvt_slices_metrics,
    plot_3d_idl_vs_correct,
    plot_gtvt_slices_metrics,
    plot_iov,
    plot_time_per_patient,
    plot_time_per_step,
    update_font_size,
)
from str_lib import DatasetPart, DatasetVer
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt

# screen cmd: /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/


# Clear Cache
if 1:
    g.clear_gpu_cache()
    g.clear_debug_data()
    g.clear_linux_trash()


# Baseline
baseline = TrainingBaseline()
baseline.new_training(
    train_remark=g.DELETE_FLAG,
    debug_mode=1,
)
# baseline.inference_on_folds(
#     baseline_id="baseline_obs.study",
#     dataset_part=DatasetPart.TEST,
#     dataset_ver=DatasetVer.OBS_STUDY,
# )
# baseline.remove_non_optimal_epochs(
#     baseline_id="baseline_2023.07.05.16.49.25",
# )
# baseline.inference_cross_valid(
#     baseline_id="baseline_obs.study",
#     dataset_part=DatasetPart.TEST,
#     dataset_ver=DatasetVer.OBS_STUDY,
# )


# IDL GTVt
# idl_gtvt = TrainingIDLGTVt()
# idl_gtvt.simulation(
#     baseline_id="baseline_simulation",
#     debug_mode=1,
# )
# idl_gtvt.inference(
#     idl_gtvt_id="idl.gtvt_2023.07.21.01.40.28",
#     dataset_part=DatasetPart.TEST,
# )
# idl_gtvt.obs_study(
#     idl_gtvt_id="idl.gtvt_" + g.DELETE_FLAG,
#     dataset_ver=DatasetVer.AU,
#     patient="106",
#     debug_mode=1,
# )


# IDL GTVn
# idl_gtvn = TrainingIDLGTVn()
# idl_gtvn.new_training(
#     baseline_id="baseline_simulation",
#     debug_mode=1,
# )
# idl_gtvn.inference_on_folds(
#     idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
#     dataset_part=DatasetPart.TEST,
#     dataset_ver=DatasetVer.AU,
# )
# idl_gtvn.remove_non_optimal_epochs(
#     idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
# )
# idl_gtvn.inference_cross_valid(
#     idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
#     dataset_ver=DatasetVer.AU,
# )
# idl_gtvn.obs_study(
#     idl_gtvn_id="idl.gtvn_" + g.DELETE_FLAG,
#     dataset_ver=DatasetVer.AU,
#     patient="106",
#     debug_mode=1,
# )


#  calculate idl gtvs simulation metric
# if 0:
#     calculate_idl_gtvs_metric(
#         idl_gtvt_id="idl.gtvt_2023.07.21.01.40.28",
#         idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
#     )


#  correction vs idl
# if 0:
#     obs_study_id_list = [
#         "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
#         "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
#         "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
#         "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
#         "idl.gtvn_2024.04.12.12.05.44_Kenneth_research",
#         "idl.gtvn_2024.04.18.11.04.48_Hanna_research",
#     ]
#     for obs_study_id in obs_study_id_list:
#         calculate_3d_idl_vs_correct(obs_study_id)

# if 0:
#     gtvn_obs_study_id_list = [
#         "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
#         "idl.gtvn_2024.04.12.12.05.44_Kenneth_research",
#         "idl.gtvn_2024.04.18.11.04.48_Hanna_research",
#     ]
#     gtvt_obs_study_id_list = [
#         "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
#         "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
#         "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
#     ]
#     for obs_study_id_list in [gtvn_obs_study_id_list, gtvt_obs_study_id_list]:
#         plot_3d_idl_vs_correct(obs_study_id_list)


#  gtvt anatomical slices
# obs_study_id_list = [
#     "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
#     "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
#     "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
# ]

# if 0:
#     for obs_study_id in obs_study_id_list:
#         calculate_gtvt_slices_metrics(obs_study_id)

# if 0:
#     for obs_study_id in obs_study_id_list:
#         calculate_gtvt_input_variation(obs_study_id)

# if 0:
#     plot_gtvt_slices_metrics(obs_study_id_list)


#  IOV
# if 0:
#     gtvt_obs_study_id_list = List(
#         (
#             "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
#             "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
#             "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
#             "label",
#         )
#     )
#     gtvn_obs_study_id_list = List(
#         (
#             "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
#             "idl.gtvn_2024.04.12.12.05.44_Kenneth_research",
#             "idl.gtvn_2024.04.18.11.04.48_Hanna_research",
#             "label",
#         )
#     )
#     for obs_study_id_list in [gtvn_obs_study_id_list, gtvt_obs_study_id_list]:
#         for pair in obs_study_id_list.get_combinations(2):
#             calculate_iov(pair[0], pair[1])

# if 0:
#     plot_iov()


#  Time consumed
# obs_study_id_list = [
#     "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
#     "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
#     "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
# ]
# if 0:
#     plot_time_per_patient(obs_study_id_list)

# if 0:
#     plot_time_per_step(obs_study_id_list)


print("Done!")
