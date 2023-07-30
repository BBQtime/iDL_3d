import sys

from custom import GPU, Debug, Directory
from custom import Global as g
from PyQt5.QtWidgets import QApplication
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt
from ui_idl import UiIdl
from ui_replay import UiReplay

GPU.clear_cache()
Debug.clear_debug_data()
Debug.clear_linux_trash()


if 1:
    app = QApplication(sys.argv)
    if 0:
        ui = UiIdl()
    else:
        ui = UiReplay()
    ui.show()
    sys.exit(app.exec_())


baseline = TrainingBaseline()
if 0:
    baseline.new_training(
        train_remark="",
        debug_mode=0,
    )


idl_gtvn = TrainingIDLGTVn()
if 0:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.05.16.49.25_unet_edge.chan=16_dropout=0.15",
        debug_mode=1,
    )


idl_gtvt = TrainingIDLGTVt()
if 0:
    idl_gtvt.simulation(
        baseline_id="baseline_2023.07.05.16.49.25_unet_edge.chan=16_dropout=0.15",
        debug_mode=1,
    )


print("Done!")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/
