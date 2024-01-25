import sys

from custom import GPU, Debug
from custom import Global as g
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


############# UI #############
if 1:
    app = QApplication(sys.argv)
    # dark theme
    app.setPalette(DarkPalette())

    if 1:
        main_win = UiIDL(
            idl_remark="",
            debug_mode=1,
        )
    else:
        main_win = UiReplay()

    main_win.show()

    # install the event filter on the QApplication instance
    # This ensures that key press events will always trigger the main window's event handler,
    # regardless of which widget currently has focus.
    app.installEventFilter(main_win)

    sys.exit(app.exec_())


############# Real IDL #############
if 0:
    idl_gtvt = TrainingIDLGTVt()
    idl_gtvt.real_idl(
        idl_gtvt_id="idl.gtvt_" + Debug.DELETE_FLAG,
        patient="106",
        dataset_ver=DatasetVer.AU_1MM,
        debug_mode=0,
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.real_idl(
        idl_gtvn_id="idl.gtvn_" + Debug.DELETE_FLAG,
        patient="106",
        dataset_part=DatasetPart.TEST_INTER,
        dataset_ver=DatasetVer.AU_1MM,
        debug_mode=0,
    )


############# Baseline #############
if 0:
    baseline = TrainingBaseline()
    baseline.new_training(
        train_remark="1mm_no.pt",
        debug_mode=1,
    )
if 0:
    baseline = TrainingBaseline()
    baseline.fold_wise_inference(
        "baseline_real.idl",
        dataset_part=DatasetPart.TEST_INTER,
        # dataset_ver=DatasetVer.MDA,
    )
if 0:
    baseline = TrainingBaseline()
    baseline.cross_valid_inference(
        "baseline_2023.07.05.16.49.25_1mm_best",
        dataset_part=DatasetPart.TEST,
        dataset_ver=DatasetVer.MDA,
    )


############# IDL.GTVt #############
if 0:
    idl_gtvt = TrainingIDLGTVt()
    idl_gtvt.simulation(
        baseline_id="baseline_real.idl",
        debug_mode=1,
    )
if 0:
    idl_gtvt = TrainingIDLGTVt()
    idl_gtvt.inference(
        "idl.gtvt_2023.08.24.16.21.09_no.pt",
        dataset_part=DatasetPart.TEST_INTER,
    )


############# IDL.GTVn #############
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.new_training(
        baseline_id="baseline_2023.07.05.16.49.25_1mm",
        debug_mode=1,
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.fold_wise_inference(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
        dataset_part=DatasetPart.TEST_INTER,
        dataset_ver=DatasetVer.AU_1MM,
    )
if 0:
    idl_gtvn = TrainingIDLGTVn()
    idl_gtvn.cross_valid_inference(
        idl_gtvn_id="idl.gtvn_2023.07.06.21.43.53",
        dataset_part=DatasetPart.TEST_INTER,
        dataset_ver=DatasetVer.AU_1MM,
    )


print("Done!")


# /home/alan/anaconda3/envs/py38/bin/python /home/alan/alan/iDL_3d/py_code/main.py

# scp -r /mnt/faststorage/alan/iDL_3d/train_results/ alan@10.60.8.15:/E:/Alan/iDL_3d/train_results/
