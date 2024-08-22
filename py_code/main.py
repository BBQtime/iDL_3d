import global_core as g
from str_lib import DatasetPart, DatasetVer, MdaObs
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt

g.clear_gpu_cache()
g.clear_linux_trash()
g.clear_debug_data()


# (1) linux cmd:
# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# (2) windows cmd
# cmd /C "C:\Users\a.wei\AppData\Local\anaconda3\envs\py39\python.exe E:\Alan\iDL_3d\py_code\main.py"

# copy from linux to windows:
# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/


baseline = TrainingBaseline()
baseline.new_training(
    train_remark="mda.transfer",
    debug_mode=0,
)
# for baseline_id, dataset_ver in [
#     ("baseline_au", DatasetVer.AU),
#     ("baseline_au_no.pt", DatasetVer.AU),
#     ("baseline_au_no.pt", DatasetVer.MDA),
#     ("baseline_mda.new", DatasetVer.MDA),
# ]:
#     baseline.inference_all_folds(
#         baseline_id=baseline_id,
#         dataset_part=DatasetPart.VALID,
#         dataset_ver=dataset_ver,
#         # debug_mode=0,
#     )
#     baseline.inference_all_folds(
#         baseline_id=baseline_id,
#         dataset_part=DatasetPart.TEST,
#         dataset_ver=dataset_ver,
#         # debug_mode=0,
#     )
#     baseline.inference_cross_valid(
#         baseline_id=baseline_id,
#         dataset_ver=dataset_ver,
#         # mda_obs=None,
#         # debug_mode=0,
#     )


# idl_gtvt = TrainingIDLGTVt()
# idl_gtvt.simulation(
#     baseline_id="baseline_au",
#     dataset_ver=DatasetVer.AU,
#     # train_remark="bias.gravity.center",
#     debug_mode=0,
# )
# idl_gtvt.inference(
#     idl_gtvt_id="",
#     debug_mode=0,
# )
# idl_gtvt.obs_study(
#     idl_gtvt_id="idl.gtvt_" + g.DELETE_FLAG,
#     dataset_ver=DatasetVer.AU,
#     patient="106",
#     debug_mode=1,
# )


# idl_gtvn = TrainingIDLGTVn()
# idl_gtvn.new_training(
#     baseline_id="baseline_mda.new",
#     debug_mode=0,
# )
# idl_gtvn.inference_all_folds(
#     idl_gtvn_id="idl.gtvn_2024.08.07.00.18.10",
#     dataset_ver=DatasetVer.MDA,
#     dataset_part=DatasetPart.VALID,
#     debug_mode=1,
# )
# idl_gtvn.inference_cross_valid(
#     idl_gtvn_id="idl.gtvn_2024.08.06.19.50.14_delete.flag",
#     dataset_ver=DatasetVer.AU,
# )
# idl_gtvn.obs_study(
#     idl_gtvn_id="idl.gtvn_" + g.DELETE_FLAG,
#     dataset_ver=DatasetVer.AU,
#     patient="106",
#     debug_mode=1,
# )


# calculate_idl_gtvs_metric(
#     idl_gtvt_id="idl.gtvt_2023.07.21.01.40.28",
#     idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
# )


# for obs_study_id in [
#     ObsStudyID.JESPER_GTVT,
#     ObsStudyID.KENNETH_GTVT,
#     ObsStudyID.HANNA_GTVT,
#     ObsStudyID.JESPER_GTVN,
#     ObsStudyID.KENNETH_GTVN,
#     ObsStudyID.HANNA_GTVN,
# ]:
#     calculate_3d_idl_vs_correct(obs_study_id)


# plot_3d_idl_vs_correct(
#     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
# )
# plot_3d_idl_vs_correct(
#     [ObsStudyID.JESPER_GTVN, ObsStudyID.KENNETH_GTVN, ObsStudyID.HANNA_GTVN]
# )


# for obs_study_id in [
#     ObsStudyID.JESPER_GTVT,
#     ObsStudyID.KENNETH_GTVT,
#     ObsStudyID.HANNA_GTVT,
# ]:
#     calculate_gtvt_slices_metrics(obs_study_id)


# for obs_study_id in [
#     ObsStudyID.JESPER_GTVT,
#     ObsStudyID.KENNETH_GTVT,
#     ObsStudyID.HANNA_GTVT,
# ]:
#     calculate_gtvt_input_variation(obs_study_id)


# plot_gtvt_slices_metrics(
#     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
# )


# for pair in List(
#     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT, "label"]
# ).get_combinations(2):
#     calculate_iov(pair[0], pair[1])


# for pair in List(
#     [ObsStudyID.JESPER_GTVN, ObsStudyID.KENNETH_GTVN, ObsStudyID.HANNA_GTVN, "label"]
# ).get_combinations(2):
#     calculate_iov(pair[0], pair[1])


# plot_iov()


# plot_time_per_patient(
#     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
# )


# plot_time_per_step(
#     [ObsStudyID.JESPER_GTVT, ObsStudyID.KENNETH_GTVT, ObsStudyID.HANNA_GTVT]
# )


print("Done!")
