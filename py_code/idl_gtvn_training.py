from baseline_training import BaselineTraining
import os
from custom import Global as g
from custom import Explorer


class IDLGTVnTraining(BaselineTraining):
    def new_training(
        self,
        baseline_id: str = None,
        baseline_fold: int = None,
        baseline_epoch: int = None,
        train_remark: str = "",
        debug_mode: bool = False,
    ):
        self._new_training(
            train_type="idl_gtvn",
            baseline_id=baseline_id,
            baseline_fold=baseline_fold,
            baseline_epoch=baseline_epoch,
            train_remark=train_remark,
            debug_mode=debug_mode,
        )

    def inference(
        self,
        idl_gtvn_id: str,
        dataset: str = "test",  # valid/test
        debug_mode: bool = False,
    ):
        if dataset != "valid":
            dataset = "test"
        self._inference(
            train_id=idl_gtvn_id,
            dataset=dataset,
            debug_mode=debug_mode,
        )

    def remove_non_optimal_epochs(
        self,
        baseline_id: str,
        baseline_fold: int = None,
        baseline_epoch: int = None,
        dataset: str = "valid",
    ):
        # find baseline fold dir
        if baseline_fold is None or baseline_fold <= 0:
            baseline_fold_dir = Explorer.get_sub_folders(
                os.path.join(g.TRAIN_RESULTS_DIR, baseline_id),
                key_word="fold=",
                return_full_path=True,
            )[0]
        else:
            baseline_fold_dir = os.path.join(
                g.TRAIN_RESULTS_DIR, baseline_id, "fold={}".format(baseline_fold)
            )
        if not os.path.exists(baseline_fold_dir):
            print("baseline fold dir not exists")
            return

        # find baseline epoch dir
        if baseline_epoch is None or baseline_epoch <= 0:
            baseline_epoch_dir = Explorer.get_sub_folders(
                baseline_fold_dir, key_word="epoch=", return_full_path=True
            )[0]
        else:
            baseline_epoch_dir = os.path.join(
                baseline_fold_dir, "epoch={:03d}".format(baseline_epoch)
            )
        if not os.path.exists(baseline_epoch_dir):
            print("baseline epoch dir not exists")
            return

        self._remove_non_optimal_epochs(
            train_results_dir=os.path.join(baseline_epoch_dir, "idl_gtvn"),
            dataset=dataset,
        )
