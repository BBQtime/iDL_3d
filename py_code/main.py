from custom import Global as g
from training_baseline import TrainingBaseline

# from training_idl_gtvt import TrainingIDLGTVt
# from training_idl_gtvn import TrainingIDLGTVn

# from training_idl_gtvs import TrainingIDLGTVs
from custom import Cleaner


Cleaner.clean_debug_data()


baseline = TrainingBaseline()
if 1:
    baseline.new_training(
        train_remark="lr=0.0001*1",
        debug_mode=0,
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


# idl_gtvn = TrainingIDLGTVn()
# if 0:
#     idl_gtvn.new_training(
#         baseline_id="baseline_2023.04.23.23.21.44_loss.gamma=0.3",
#         debug_mode=1,
#     )
# if 0:
#     idl_gtvn.inference(
#         "idl_gtvn_2023.05.26.12.06.15_distance.map_4modals_augment_unified.focal.loss_unet.pp.slim",
#         dataset="test.inter",
#     )


# idl = TrainingIDLGTVs()
# if 0:
#     idl.new_training(
#         "baseline_2023.02.27.07.08.09_loss.gamma=0.5",
#         # train_remark=g.DELETE_FLAG,
#         debug_mode=0,
#     )
# if 0:
#     idl.inference("idl_2023.06.05.03.53.29")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/

print("Done!")
