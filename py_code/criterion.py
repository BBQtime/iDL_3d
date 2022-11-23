import global_elems as g
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy import ndarray
from typing import Union, Tuple


SMOOTH = 0.000001
# EPSILON = torch.finfo(torch.float32).eps


def __3d_tversky(
    preds: Tensor, labels: Tensor, fp_weight: float
) -> Tuple[Tensor, Tensor]:
    # fp_weight=0.5, this function becomes dsc
    fn_weight = 1 - fp_weight
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    tn = ((1 - preds) * (1 - labels)).sum()
    fore_score = (tp + SMOOTH) / (tp + fp_weight * fp + fn_weight * fn + SMOOTH)
    back_score = (tn + SMOOTH) / (tn + fp_weight * fp + fn_weight * fn + SMOOTH)
    return fore_score, back_score


def __asymmetric_focal_tversky_loss(
    preds: Tensor,
    labels: Tensor,
    fore_weight: float,
    fore_power: float,
    fp_weight: float,
) -> Tensor:
    # fore_power=1, this function becomes "focal_tversky_loss"
    fore_score, back_score = __3d_tversky(
        preds=preds, labels=labels, fp_weight=fp_weight
    )
    back_loss = 1 - back_score
    fore_loss = 1 - fore_score
    # enhancing foreground
    fore_loss = torch.pow(fore_loss, fore_power)
    return fore_weight * fore_loss + (1 - fore_weight) * back_loss


def __3d_bce(preds: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    fore_bce = F.binary_cross_entropy(input=preds, target=labels, reduction="mean")
    back_bce = F.binary_cross_entropy(
        input=(1 - preds), target=(1 - labels), reduction="mean"
    )
    return fore_bce, back_bce


def __asymmetric_focal_loss(
    preds: Tensor, labels: Tensor, fore_weight: float, back_power: float
) -> Tensor:
    # Focal loss is used to address the issue of the class imbalance problem.
    # power > 1: depress easy examples
    fore_bce, back_bce = __3d_bce(preds=preds, labels=labels)
    # depress back bce
    back_bce = torch.pow(1 - preds, back_power) * back_bce
    back_bce = torch.mean(back_bce)
    # back_bce_exp = torch.exp(-back_bce)
    # back_bce = torch.pow(1 - back_bce_exp, back_power) * back_bce
    return fore_weight * fore_bce + (1 - fore_weight) * back_bce


def _hybrid_focal_loss(
    preds: Tensor,
    labels: Tensor,
    hybrid_weight: float,
    tversky_fore_weight: float,
    tversky_fore_power: float,
    tversky_fp_weight: float,
    bce_fore_weight: float,
    bce_back_power: float,
) -> Tensor:
    # hybrid_weight: weight given to focal_tversky_loss and focal_loss
    focal_tversky_loss = __asymmetric_focal_tversky_loss(
        preds=preds,
        labels=labels,
        fore_weight=tversky_fore_weight,
        fore_power=tversky_fore_power,
        fp_weight=tversky_fp_weight,
    )
    focal_loss = __asymmetric_focal_loss(
        preds=preds,
        labels=labels,
        fore_weight=bce_fore_weight,
        back_power=bce_back_power,
    )
    return hybrid_weight * focal_tversky_loss + (1 - hybrid_weight) * focal_loss


# only for training
class HybridFocalLoss(nn.Module):
    def __init__(
        self,
        hybrid_weight: float = 0.5,
        tversky_fore_weight: float = 0.5,
        tversky_fore_power: float = 0.35,
        tversky_fp_weight: float = 0.5,
        bce_fore_weight: float = 0.5,
        bce_back_power: float = 2.0,
    ):
        super().__init__()

        self.hybrid_weight = hybrid_weight
        self.tversky_fore_weight = tversky_fore_weight
        self.tversky_fore_power = tversky_fore_power
        self.tversky_fp_weight = tversky_fp_weight
        self.bce_fore_weight = bce_fore_weight
        self.bce_back_power = bce_back_power

    def forward(self, preds, labels):
        if preds.shape != labels.shape:
            g.exit_app("UnifiedFocalLoss(): preds.shape != labels.shape")

        return _hybrid_focal_loss(
            preds=preds,
            labels=labels,
            hybrid_weight=self.hybrid_weight,
            tversky_fore_weight=self.tversky_fore_weight,
            tversky_fore_power=self.tversky_fore_power,
            tversky_fp_weight=self.tversky_fp_weight,
            bce_fore_weight=self.bce_fore_weight,
            bce_back_power=self.bce_back_power,
        )
