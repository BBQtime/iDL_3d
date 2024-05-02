import global_core as g
from research_analysis import (
    calculate_metrics_correct_vs_idl,
    calculate_metrics_gtvt_delineation_vs_idl,
    create_table_correct_vs_idl,
    create_table_gtvt_delineation_vs_idl,
    plot_fig_correct_vs_idl,
)

# from str_lib import DatasetPart, DatasetVer
# from training_baseline import TrainingBaseline
# from training_idl_gtvn import TrainingIDLGTVn
# from training_idl_gtvt import TrainingIDLGTVt

# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/


############# Clear Cache #############
if 0:
    g.clear_gpu_cache()
    g.clear_debug_data()
    g.clear_linux_trash()


############# correction vs idl #############
if 1:
    obs_study_id_list = [
        "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
        "idl.gtvn_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvn_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvn_2024.04.18.11.04.48_Hanna_research",
    ]

    if 0:
        for obs_study_id in obs_study_id_list:
            calculate_metrics_correct_vs_idl(obs_study_id)

    if 0:
        create_table_correct_vs_idl(obs_study_id_list)

    if 1:
        plot_fig_correct_vs_idl(obs_study_id_list)


############# gtvt delineation vs idl #############
if 1:
    obs_study_id_list = [
        "idl.gtvt_2024.03.18.09.05.54_Jesper_research",
        "idl.gtvt_2024.04.12.12.05.44_Kenneth_research",
        "idl.gtvt_2024.04.18.11.04.48_Hanna_research",
    ]

    if 0:
        for obs_study_id in obs_study_id_list:
            calculate_metrics_gtvt_delineation_vs_idl(obs_study_id)

    if 0:
        create_table_gtvt_delineation_vs_idl(obs_study_id_list)


# ############# Baseline #############
# if 0:
#     baseline = TrainingBaseline()
#     baseline.new_training(
#         train_remark=g.DELETE_FLAG,
#         debug_mode=1,
#     )
# if 0:
#     baseline = TrainingBaseline()
#     baseline.inference_on_folds(
#         baseline_id="baseline_obs.study",
#         dataset_part=DatasetPart.TEST,
#         dataset_ver=DatasetVer.OBS_STUDY,
#     )
# if 0:
#     baseline = TrainingBaseline()
#     baseline.remove_non_optimal_epochs(
#         baseline_id="baseline_2023.07.05.16.49.25",
#     )
# if 0:
#     baseline = TrainingBaseline()
#     baseline.inference_cross_valid(
#         baseline_id="baseline_obs.study",
#         dataset_part=DatasetPart.TEST,
#         dataset_ver=DatasetVer.OBS_STUDY,
#     )


# ############# IDL.GTVt #############
# if 0:
#     idl_gtvt = TrainingIDLGTVt()
#     idl_gtvt.simulation(
#         baseline_id="baseline_simulation",
#         debug_mode=1,
#     )
# if 0:
#     idl_gtvt = TrainingIDLGTVt()
#     idl_gtvt.inference(
#         idl_gtvt_id="idl.gtvt_2023.07.21.01.40.28",
#         dataset_part=DatasetPart.TEST,
#     )


# ############# IDL.GTVn #############
# if 0:
#     idl_gtvn = TrainingIDLGTVn()
#     idl_gtvn.new_training(
#         baseline_id="baseline_simulation",
#         debug_mode=1,
#     )
# if 0:
#     idl_gtvn = TrainingIDLGTVn()
#     idl_gtvn.inference_on_folds(
#         idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
#         dataset_part=DatasetPart.TEST,
#         dataset_ver=DatasetVer.AU,
#     )
# if 0:
#     idl_gtvn = TrainingIDLGTVn()
#     idl_gtvn.remove_non_optimal_epochs(
#         idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
#     )
# if 0:
#     idl_gtvn = TrainingIDLGTVn()
#     idl_gtvn.inference_cross_valid(
#         idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
#         dataset_ver=DatasetVer.AU,
#     )


# ############# Observer Study #############
# if 0:
#     idl_gtvt = TrainingIDLGTVt()
#     idl_gtvt.obs_study(
#         idl_gtvt_id="idl.gtvt_" + g.DELETE_FLAG,
#         dataset_ver=DatasetVer.AU,
#         patient="106",
#         debug_mode=1,
#     )
# if 0:
#     idl_gtvn = TrainingIDLGTVn()
#     idl_gtvn.obs_study(
#         idl_gtvn_id="idl.gtvn_" + g.DELETE_FLAG,
#         dataset_ver=DatasetVer.AU,
#         patient="106",
#         debug_mode=1,
#     )


print("Done!")
