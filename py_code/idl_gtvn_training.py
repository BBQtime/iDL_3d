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

    def remove_non_optimal_epochs(self, idl_gtvn_id: str, dataset: str = "valid"):
        self._remove_non_optimal_epochs(train_id=idl_gtvn_id, dataset=dataset)
