from training_baseline import TrainingBaseline
from training_idl_gtvt import TrainingIDLGTVt
from training_idl_gtvn import TrainingIDLGTVn
from custom import Cleaner


Cleaner.clean_debug_data()


baseline = TrainingBaseline()
if 0:
    baseline.new_training(
        train_remark="",
        debug_mode=1,
    )
if 0:
    baseline.inference(
        "baseline_2023.04.23.23.21.44_loss.gamma=0.3",
        dataset="valid",
    )


idl_gtvt = TrainingIDLGTVt()
if 0:
    idl_gtvt.simulation(
        "baseline_2023.04.23.23.21.44_loss.gamma=0.3",
        train_remark="",
        debug_mode=1,
    )
if 0:
    idl_gtvt.inference("idl_gtvt_2023.04.04.14.43.54_fp.fn:4.0")


idl_gtvn = TrainingIDLGTVn()
if 0:
    idl_gtvn.new_training(
        baseline_id="baseline_2023.04.23.23.21.44_loss.gamma=0.3",
        debug_mode=1,
    )
if 0:
    idl_gtvn.inference(
        "idl_gtvn_2023.05.26.12.06.15_distance.map_4modals_augment_unified.focal.loss_unet.pp.slim",
        dataset="test",
    )

print("Done!")
