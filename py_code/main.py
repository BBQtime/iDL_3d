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


############# UI #############
if 0:
    app = QApplication(sys.argv)
    if 0:
        ui = UiIdl(debug_mode=1)
    else:
        ui = UiReplay()
    ui.show()
    sys.exit(app.exec_())


############# Baseline #############
baseline = TrainingBaseline()
if 1:
    baseline.new_training(
        train_remark="",
        debug_mode=1,
    )
if 0:
    baseline.inference(
        # "baseline_2023.07.05.16.49.25_1mm_best",
        "baseline_2023.02.27.07.08.09_3mm_best",
        dataset_section="test.inter",
    )
if 0:
    baseline.cross_valid_evaluation(
        # "baseline_2023.07.05.16.49.25_1mm_best",
        "baseline_2023.02.27.07.08.09_3mm_best",
        dataset_section="test.inter",
    )


############# IDL.GTVn #############
idl_gtvn = TrainingIDLGTVn()
if 1:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.05.16.49.25_1mm_best",
        train_remark="",
        debug_mode=1,
    )
if 0:
    idl_gtvn.inference(
        "idl.gtvn_2023.07.06.21.43.53",
        # "idl.gtvn_2023.05.26.12.06.15_best",
        dataset_section="test.inter",
    )
if 0:
    idl_gtvn.cross_valid_evaluation(
        "idl.gtvn_2023.07.06.21.43.53",
        # "idl.gtvn_2023.05.26.12.06.15_best",
        dataset_section="test.inter",
    )


############# IDL.GTVt #############
idl_gtvt = TrainingIDLGTVt()
if 0:
    idl_gtvt.simulation(
        baseline_id="baseline_2023.07.05.16.49.25_1mm_best",
        debug_mode=0,
    )


print("Done!")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/
