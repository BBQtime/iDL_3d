from custom import GPU, Debug
from custom import Global as g
from str_lib import DatasetPart, DatasetVer
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt

if 0:
    GPU.clear_cache()
    Debug.clear_debug_data()
    Debug.clear_linux_trash()


############# Observer Study #############
if 0:
    idl_gtvt = TrainingIDLGTVt()
    idl_gtvt.obs_study(
        idl_gtvt_id="idl.gtvt_" + Debug.DELETE_FLAG,
        patient="106",
        debug_mode=1,
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.obs_study(
        idl_gtvn_id="idl.gtvn_" + Debug.DELETE_FLAG,
        patient="106",
        debug_mode=1,
    )


############# Baseline #############
if 0:
    baseline = TrainingBaseline()
    baseline.new_training(
        train_remark="obs.study.baseline",
        debug_mode=1,
    )
if 0:
    baseline = TrainingBaseline()
    baseline.inference_on_folds(
        baseline_id="baseline_obs.study",
        dataset_part=DatasetPart.TEST,
        dataset_ver=DatasetVer.OBS_STUDY,
    )
if 0:
    baseline = TrainingBaseline()
    baseline.remove_non_optimal_epochs(
        baseline_id="baseline_2023.07.05.16.49.25",
    )
if 1:
    baseline = TrainingBaseline()
    baseline.inference_cross_valid(
        baseline_id="baseline_obs.study",
        dataset_part=DatasetPart.TEST,
        dataset_ver=DatasetVer.OBS_STUDY,
    )


############# IDL.GTVt #############
if 0:
    idl_gtvt = TrainingIDLGTVt()
    idl_gtvt.simulation(
        baseline_id="baseline_2023.07.05.16.49.25",
        debug_mode=1,
    )
if 0:
    idl_gtvt = TrainingIDLGTVt()
    idl_gtvt.inference(
        idl_gtvt_id="idl.gtvt_2023.07.21.01.40.28",
        dataset_part=DatasetPart.TEST,
    )


############# IDL.GTVn #############
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.05.16.49.25",
        debug_mode=1,
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.inference_on_folds(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
        dataset_part=DatasetPart.TEST,
        dataset_ver=DatasetVer.AU,
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.remove_non_optimal_epochs(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.inference_cross_valid(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
        dataset_ver=DatasetVer.AU,
    )


print("Done!")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/
