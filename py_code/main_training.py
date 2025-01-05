import global_utils.global_core as g
from dataset_utils import label_preprocess
from global_utils.str_lib import DatasetPart, DatasetVer
from training_utils.baseline_training import BaselineTraining
from training_utils.idl_gtvn_training import IDLGTVnTraining
from training_utils.idl_gtvt_training import IDLGTVtTraining

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

    # for dataset_ver in [
    #     # DatasetVer.AU,
    #     # DatasetVer.AU_EXT, # done, 876
    #     # DatasetVer.OBS_STUDY,  # no gtvn fregment
    #     # DatasetVer.MDA,  # no gtvn fregment
    #     # DatasetVer.NKI, # done, 292/296
    # ]:
    #     print(f"Dataset: {dataset_ver}")
    #     for fregment_threshold in [2, 5, 10]:
    #         label_preprocess.remove_label_fregments(
    #             dataset_ver=dataset_ver,
    #             fregment_threshold=fregment_threshold,
    #             debug_mode=1,
    #         )

    # for dataset_ver in [
    #     # DatasetVer.AU,
    #     # DatasetVer.AU_EXT,  # done, 876
    #     # DatasetVer.OBS_STUDY,  # no need to update
    #     # DatasetVer.MDA,
    #     # DatasetVer.NKI,
    # ]:
    #     label_preprocess.generate_gtvn_clicks_nii(dataset_ver)
    #     label_preprocess.check_gtvn_clicks_within_label(dataset_ver)

    # baseline = BaselineTraining()
    # # baseline.new_training(
    # #     # train_remark="mda.transfer",
    # #     device_id=-1,
    # #     debug_mode=1,
    # # )
    # for baseline_id, dataset_ver in [
    #     ("baseline_au", DatasetVer.AU_EXT),
    #     ("baseline_au_no.pt", DatasetVer.AU_EXT),
    # ]:
    #     device_id = 1
    #     debug_mode = 0
    #     # baseline.inference_all_folds(
    #     #     baseline_id=baseline_id,
    #     #     dataset_ver=dataset_ver,
    #     #     dataset_part=DatasetPart.VALID,
    #     #     device_id=device_id,
    #     #     debug_mode=debug_mode,
    #     # )
    #     baseline.inference_all_folds(
    #         baseline_id=baseline_id,
    #         dataset_ver=dataset_ver,
    #         dataset_part=DatasetPart.TEST,
    #         device_id=device_id,
    #         debug_mode=debug_mode,
    #     )
    #     baseline.inference_cross_valid(
    #         baseline_id=baseline_id,
    #         dataset_ver=dataset_ver,
    #         device_id=device_id,
    #         debug_mode=debug_mode,
    #     )

    # idl_gtvn = IDLGTVnTraining()
    # for baseline_id in [
    #     # "baseline_au",
    #     # "baseline_au_no.pt",
    #     # "baseline_mda.new",
    #     # "baseline_mda.transfer",
    #     # "baseline_nki.new",
    #     # "baseline_nki.transfer",
    # ]:
    #     idl_gtvn.new_training(
    #         baseline_id=baseline_id,
    #         train_remark=baseline_id[len("baseline_") :] + "_multi.clicks",
    #         device_id=1,
    #         debug_mode=0,
    #     )
    # for idl_gtvn_id, dataset_ver in [
    #     ("idl.gtvn_mda.new_multi.clicks", DatasetVer.MDA),
    #     ("idl.gtvn_au_multi.clicks", DatasetVer.AU_EXT),
    #     ("idl.gtvn_au_no.pt_multi.clicks", DatasetVer.AU_EXT),
    # ]:
    #     device_id = 1
    #     debug_mode = 0
    #     if idl_gtvn_id == "idl.gtvn_mda.new_multi.clicks":
    #         idl_gtvn.inference_all_folds(
    #             idl_gtvn_id=idl_gtvn_id,
    #             dataset_ver=dataset_ver,
    #             dataset_part=DatasetPart.VALID,
    #             device_id=device_id,
    #             debug_mode=debug_mode,
    #         )
    #     idl_gtvn.inference_all_folds(
    #         idl_gtvn_id=idl_gtvn_id,
    #         dataset_ver=dataset_ver,
    #         dataset_part=DatasetPart.TEST,
    #         device_id=device_id,
    #         debug_mode=debug_mode,
    #     )
    #     idl_gtvn.inference_cross_valid(
    #         idl_gtvn_id=idl_gtvn_id,
    #         dataset_ver=dataset_ver,
    #         device_id=device_id,
    #         debug_mode=debug_mode,
    #     )
    # idl_gtvn.obs_study(
    #     idl_gtvn_id="idl.gtvn_" + g.DELETE_FLAG,
    #     dataset_ver=DatasetVer.OBS_STUDY,
    #     patient="489",
    #     device_id=1,
    #     debug_mode=1,
    # )

    idl_gtvt = IDLGTVtTraining()
    for baseline_id in [
        "baseline_au",
        # "baseline_au_no.pt",
    ]:
        idl_gtvt.simulation(
            baseline_id=baseline_id,
            device_id=0,
            train_remark=DatasetVer.AU_EXT,
            debug_mode=0,
        )
    # idl_gtvt.inference(
    #     idl_gtvt_id="idl.gtvt_obs.study_bug",
    #     debug_mode=0,
    # )
    # idl_gtvt.obs_study(
    #     idl_gtvt_id="idl.gtvt_" + g.DELETE_FLAG,
    #     dataset_ver=DatasetVer.OBS_STUDY,
    #     patient="489",
    #     device_id=1,
    #     debug_mode=1,
    # )

    print("Done!")
