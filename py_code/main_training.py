import global_utils.global_core as g
from global_utils.str_lib import DatasetPart, DatasetVer
from training_utils.training_baseline import TrainingBaseline
from training_utils.training_idl_gtvn import TrainingIDLGTVn
from training_utils.training_idl_gtvt import TrainingIDLGTVt

# (1) linux cmd:
# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main_training.py

# (2) windows cmd
# cmd /C "C:\Users\a.wei\AppData\Local\anaconda3\envs\py39\python.exe E:\Alan\iDL_3d\py_code\main_training.py"

# copy results from linux to windows:
# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/


if __name__ == "__main__":

    g.clear_gpu_cache()
    g.clear_linux_trash()
    g.clear_debug_data()

    # baseline = TrainingBaseline()
    # baseline.new_training(
    #     # train_remark="mda.transfer",
    #     debug_mode=1,
    # )
    # for baseline_id, dataset_ver in [
    #     ("baseline_au_no.pt", DatasetVer.AU_EXT),
    #     ("baseline_au", DatasetVer.AU_EXT),
    # ]:
    #     debug_mode = 0
    # baseline.inference_all_folds(
    #     baseline_id=baseline_id,
    #     dataset_part=DatasetPart.VALID,
    #     dataset_ver=dataset_ver,
    #     debug_mode=debug_mode,
    # )
    # baseline.inference_all_folds(
    #     baseline_id=baseline_id,
    #     dataset_part=DatasetPart.TEST,
    #     dataset_ver=dataset_ver,
    #     debug_mode=debug_mode,
    # )
    # baseline.inference_cross_valid(
    #     baseline_id=baseline_id,
    #     dataset_ver=dataset_ver,
    #     debug_mode=debug_mode,
    # )

    # idl_gtvn = TrainingIDLGTVn()
    # idl_gtvn.new_training(
    #     baseline_id="baseline_au",
    #     train_remark="nki.transfer",
    #     debug_mode=1,
    # )
    # for idl_gtvn_id, dataset_ver in [
    #     ("idl.gtvn_au_no.pt", DatasetVer.AU_EXT),
    #     ("idl.gtvn_au", DatasetVer.AU_EXT),
    # ]:
    #     debug_mode = 0
    # idl_gtvn.inference_all_folds(
    #     idl_gtvn_id=idl_gtvn_id,
    #     dataset_ver=dataset_ver,
    #     dataset_part=DatasetPart.VALID,
    #     debug_mode=debug_mode,
    # )
    # idl_gtvn.inference_all_folds(
    #     idl_gtvn_id=idl_gtvn_id,
    #     dataset_ver=dataset_ver,
    #     dataset_part=DatasetPart.TEST,
    #     debug_mode=debug_mode,
    # )
    # idl_gtvn.inference_cross_valid(
    #     idl_gtvn_id=idl_gtvn_id,
    #     dataset_ver=dataset_ver,
    #     debug_mode=debug_mode,
    # )
    # idl_gtvn.obs_study(
    #     idl_gtvn_id="idl.gtvn_" + g.DELETE_FLAG,
    #     dataset_ver=DatasetVer.AU,
    #     patient="106",
    #     debug_mode=1,
    # )

    # idl_gtvt = TrainingIDLGTVt()
    # for baseline_id in [
    #     "baseline_au",
    # ]:
    #     idl_gtvt.simulation(
    #         baseline_id=baseline_id,
    #         # train_remark="au.ext_bug.fixed",
    #         debug_mode=1,
    #     )
    # idl_gtvt.inference(
    #     idl_gtvt_id="idl.gtvt_obs.study_bug",
    #     debug_mode=0,
    # )
    # idl_gtvt.obs_study(
    #     idl_gtvt_id="idl.gtvt_" + g.DELETE_FLAG,
    #     dataset_ver=DatasetVer.AU,
    #     patient="106",
    #     debug_mode=1,
    # )

    print("Done!")
