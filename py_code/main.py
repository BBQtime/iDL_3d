from custom import Global as g
from custom import Debug
from custom import Explorer
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn

# from training_idl_gtvt import TrainingIDLGTVt


Debug.clean_gpu_cache()
Debug.clean_debug_data()
Debug.clean_linux_trash()


baseline = TrainingBaseline()
if 0:
    baseline.new_training(
        train_remark="unet_edge.chan=16_dropout=0.3",
        debug_mode=1,
    )
if 1:
    baseline_id_list = Explorer.get_sub_folders(
        g.TRAIN_RESULTS_DIR, "baseline_", full_path=False
    )
    for baseline_id in baseline_id_list:
        baseline.inference(baseline_id)
        baseline.remove_non_optimal_epochs(baseline_id)
        baseline.calculate_cross_valid_scores(baseline_id)


idl_gtvn = TrainingIDLGTVn()
if 0:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.06.00.17.59_unet_edge.chan=16_dropout=0.3",
        debug_mode=1,
    )
if 1:
    for idl_id in ["idl.gtvn_2023.05.26.12.06.15_best", "idl.gtvn_2023.07.02.21.43.53"]:
        idl_gtvn.inference(idl_id)
        idl_gtvn.remove_non_optimal_epochs(idl_id)
        idl_gtvn.calculate_cross_valid_scores(idl_id)


# idl_gtvt = TrainingIDLGTVt()
# if 0:
#     idl_gtvt.simulation(
#         "baseline_2023.04.23.23.21.44_loss.gamma=0.3",
#         train_remark="",
#         debug_mode=0,
#     )
# if 0:
#     idl_gtvt.inference("idl_gtvt_2023.04.04.14.43.54")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/

print("Done!")
