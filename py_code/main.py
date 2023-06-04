from baseline_training import BaselineTraining
from idl_gtvt_training import IDLGTVtTraining
from idl_gtvn_training import IDLGTVnTraining
from custom import Cleaner

Cleaner.clean_debug_data()


baseline_training = BaselineTraining()
if 0:
    baseline_training.new_training(debug_mode=1)
if 1:
    baseline_training.inference(
        "baseline_2023.04.23.23.21.44_loss.gamma=0.3",
        dataset="test",
    )


idl_gtvt_training = IDLGTVtTraining()
if 0:
    idl_gtvt_training.simulation(
        "baseline_2023.04.23.23.21.44_loss.gamma=0.3",
        train_remark="fp.fn:4.0",
        debug_mode=0,
    )


idl_gtvn_training = IDLGTVnTraining()
if 0:
    idl_gtvn_training.inference(
        "idl_gtvn_2023.05.26.12.06.15_distance.map_4modals_augment_unified.focal.loss_unet.pp.slim_post.process",
        dataset="test",
    )

print("Done!")
