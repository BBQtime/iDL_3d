import sys

from custom import Debug, Explorer
from custom import Global as g
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt

Debug.clean_gpu_cache()
Debug.clean_debug_data()
Debug.clean_linux_trash()


baseline = TrainingBaseline()
if 0:
    baseline.new_training(
        train_remark="unet_edge.chan=16_dropout=0.3",
        debug_mode=1,
    )


idl_gtvn = TrainingIDLGTVn()
if 0:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.05.16.49.25_unet_edge.chan=16_dropout=0.15",
        debug_mode=1,
    )


idl_gtvt = TrainingIDLGTVt()
if 1:
    idl_gtvt.simulation(
        baseline_id="baseline_2023.07.05.16.49.25_unet_edge.chan=16_dropout=0.15",
        train_remark="",
        debug_mode=1,
    )
if 0:
    idl_gtvt.inference("idl_gtvt_2023.07.16.01.50.07_best")


print("Done!")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/
