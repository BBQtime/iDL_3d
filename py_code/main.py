import sys

from custom import GPU, Debug
from darktheme.widget_template import DarkPalette
from PyQt5.QtWidgets import QApplication
from str_lib import DatasetPart, DatasetVer
from training_baseline import TrainingBaseline
from training_idl_gtvn import TrainingIDLGTVn
from training_idl_gtvt import TrainingIDLGTVt
from ui_idl import UiIDL
from ui_replay import UiReplay

GPU.clear_cache()
Debug.clear_debug_data()
Debug.clear_linux_trash()


# progress bar: multi thread
# list of idl step


############# UI #############
if 1:
    # app = DarkApplication(sys.argv)
    app = QApplication(sys.argv)
    app.setPalette(DarkPalette())

    if 1:
        main_win = UiIDL(
            idl_remark="",
            debug_mode=1,
        )
    else:
        main_win = UiReplay()

    main_win.show()
    sys.exit(app.exec_())


############# Baseline #############
baseline = TrainingBaseline()
if 0:
    baseline.new_training(
        train_remark="1mm_no.pt",
        debug_mode=1,
    )
if 0:
    baseline.fold_wise_inference(
        "baseline_real.idl",
        dataset_part=DatasetPart.TEST_INTER,
        # dataset_ver=DatasetVer.MDA,
    )
if 0:
    baseline.cross_valid_inference(
        "baseline_2023.07.05.16.49.25_1mm_best",
        dataset_part=DatasetPart.TEST,
        dataset_ver=DatasetVer.MDA,
    )


############# IDL.GTVn #############
idl_gtvn = TrainingIDLGTVn()
if 0:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.05.16.49.25_1mm",
        # train_remark="fold.5",
        debug_mode=1,
    )
if 0:
    idl_gtvn.fold_wise_inference(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
        dataset_part=DatasetPart.TEST_INTER,
        dataset_ver=DatasetVer.AU_1MM,
    )
if 0:
    idl_gtvn.cross_valid_inference(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
        dataset_part=DatasetPart.TEST_INTER,
        dataset_ver=DatasetVer.AU_1MM,
    )
if 0:
    idl_gtvn.real_idl(
        idl_gtvn_id="idl.gtvn_test",
        patient="106",
        dataset_part=DatasetPart.TEST_INTER,
        dataset_ver=DatasetVer.AU_1MM,
    )


############# IDL.GTVt #############
idl_gtvt = TrainingIDLGTVt()
if 0:
    idl_gtvt.new_training(
        baseline_id="baseline_2023.07.05.16.49.25_1mm",
        debug_mode=1,
    )
if 0:
    idl_gtvt.inference(
        "idl.gtvt_2023.08.24.16.21.09_no.pt",
        dataset_part=DatasetPart.TEST_INTER,
    )


print("Done!")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/
