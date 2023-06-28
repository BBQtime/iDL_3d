from custom import Global as g
from custom import Cleaner
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn

# from training_idl_gtvt import TrainingIDLGTVt


Cleaner.clean_debug_data()


baseline = TrainingBaseline()
if 0:
    baseline.new_training(
        train_remark="unet_edge.chan=16",
        debug_mode=1,
    )


idl_gtvn = TrainingIDLGTVn()
if 1:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.02.27.07.08.09_3mm_best",
        debug_mode=1,
    )


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
