import global_elems as g
import chart_maker
from tqdm import tqdm
from baseline_training import BaselineTraining
from idl_training import IDLTraining

baseline = BaselineTraining()
idl = IDLTraining()

# baseline start new trainings
if 1:
    baseline_id_list = baseline.train(
        # train_remark="test",
        # debug_mode=1,
    )

# baseline test exist cnns' scores
if 0:
    baseline.inference("2022.08.28.23.34.56_2022.08.28.23.34.56_debug.mode.delete.this")


# simulated iDL
if 0:
    idl.simulation(
        # train_remark="",
        debug_mode=1,
    )


# baseline results visualization
if 0:
    idl.baseline_visualize(
        baseline_id="2022.03.19.23.16.31_2022.03.19.23.16.31_unet++_dropout=0.3_optimal",
        idl_results_folder=g.IDL_RESULTS_FOLDER,
        idl_id="baseline.visualize",
    )

# real-life iDL training
if 0:
    idl.real_training(
        baseline_id="2022.03.19.23.16.31_2022.03.19.23.16.31_unet++_dropout=0.3",
        idl_results_folder="F:/",
        idl_id="iDL",
        cur_patient="106",
        cur_round=1,
    )

if 0:
    chart_maker.compare_idl_results(
        key_hyper="select.step",
        idl_id_list=g.get_sub_folders(
            g.IDL_RESULTS_FOLDER,
            "select.step=",
        ),
    )

if 0:
    chart_maker.patients_overview(
        "2022.03.25.15.07.51_2022.03.25.15.07.40_lr=5e-06,5e-06_optimal"
    )


g.print_line()
print("success")
