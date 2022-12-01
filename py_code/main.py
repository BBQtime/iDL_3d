import global_elems as g
import chart_maker
from tqdm import tqdm
import os
from baseline_training import BaselineTraining
from idl_training import IDLTraining


# /home/alan/anaconda3/envs/py38/bin/python /mnt/faststorage/alan/iDL_3d/py_code/main.py


baseline = BaselineTraining()
idl = IDLTraining()


# baseline:
# patch.tumor.size.pct
# patch.non.empty.pct

# idl:
# patch.ignore.other.slices

# # baseline start new trainings
# if 0:
#     baseline.training(
#         # train_remark="sym.unified.focal.loss",
#         train_remark="target.vol.pct=0",
#         debug_mode=0,
#     )


# simulated iDL
if 1:
    idl.simulation(
        baseline_id="baseline_2022.11.27.06.23.46_target.vol.pct=0_lr=0.0005",
        train_remark="delete.this",
        debug_mode=0,
    )


# # real-life iDL training
# if 0:
#     idl.real_training(
#         baseline_id="2022.03.19.23.16.31_2022.03.19.23.16.31_unet++_dropout=0.3",
#         idl_results_folder="F:/",
#         idl_id="iDL",
#         cur_patient="106",
#         cur_round=1,
#     )


# if 0:
#     chart_maker.compare_idl_results(
#         key_hyper="select.step",
#         idl_id_list=g.get_sub_folders(
#             g.IDL_RESULTS_FOLDER,
#             "select.step=",
#         ),
#     )


# if 0:
#     chart_maker.patients_overview(
#         "2022.03.25.15.07.51_2022.03.25.15.07.40_lr=5e-06,5e-06_optimal"
#     )


g.print_line()
print("success")
