import global_elems as g
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import surface_distance as sd
from torch import Tensor
from numpy import ndarray
from typing import Union


SMOOTH = 0.000001
# EPSILON = torch.finfo(torch.float32).eps


def __2d_tversky(
    preds: Tensor, labels: Tensor, fp_weight: float
) -> tuple[Tensor, Tensor]:
    # fp_weight=0.5, this function becomes dsc
    fn_weight = 1 - fp_weight
    depth = labels.size(0)
    fore_score = 0
    back_score = 0
    for i in range(depth):
        tp = (preds[i] * labels[i]).sum()
        fp = (preds[i] * (1 - labels[i])).sum()
        fn = ((1 - preds[i]) * labels[i]).sum()
        tn = ((1 - preds[i]) * (1 - labels[i])).sum()
        fore_score += (tp + SMOOTH) / (tp + fp_weight * fp + fn_weight * fn + SMOOTH)
        back_score += (tn + SMOOTH) / (tn + fp_weight * fp + fn_weight * fn + SMOOTH)
    return fore_score / depth, back_score / depth


def __3d_tversky(
    preds: Tensor, labels: Tensor, fp_weight: float
) -> tuple[Tensor, Tensor]:
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
    dim: str,
    fore_weight: float,
    fore_power: float,
    fp_weight: float,
) -> Tensor:
    # fore_power=1, this function becomes "focal_tversky_loss"
    if dim == "2d":
        fore_score, back_score = __2d_tversky(
            preds=preds, labels=labels, fp_weight=fp_weight
        )
    else:
        fore_score, back_score = __3d_tversky(
            preds=preds, labels=labels, fp_weight=fp_weight
        )
    back_loss = 1 - back_score
    fore_loss = 1 - fore_score
    # enhancing foreground
    fore_loss = torch.pow(fore_loss, fore_power)
    return fore_weight * fore_loss + (1 - fore_weight) * back_loss


def __2d_bce(preds: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    depth = labels.size(0)
    fore_bce = 0
    back_bce = 0
    for i in range(depth):
        fore_bce += F.binary_cross_entropy(
            input=preds[i], target=labels[i], reduction="mean"
        )
        back_bce += F.binary_cross_entropy(
            input=1 - preds[i], target=1 - labels[i], reduction="mean"
        )
    return fore_bce / depth, back_bce / depth


def __3d_bce(preds: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    fore_bce = F.binary_cross_entropy(input=preds, target=labels, reduction="mean")
    back_bce = F.binary_cross_entropy(
        input=(1 - preds), target=(1 - labels), reduction="mean"
    )
    return fore_bce, back_bce


def __asymmetric_focal_loss(
    preds: Tensor, labels: Tensor, dim: str, fore_weight: float, back_power: float
) -> Tensor:
    # Focal loss is used to address the issue of the class imbalance problem.
    # power > 1: depress easy examples
    if dim == "2d":
        fore_bce, back_bce = __2d_bce(preds=preds, labels=labels)
    else:
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
    dim: str,
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
        dim=dim,
        fore_weight=tversky_fore_weight,
        fore_power=tversky_fore_power,
        fp_weight=tversky_fp_weight,
    )
    focal_loss = __asymmetric_focal_loss(
        preds=preds,
        labels=labels,
        dim=dim,
        fore_weight=bce_fore_weight,
        back_power=bce_back_power,
    )
    return hybrid_weight * focal_tversky_loss + (1 - hybrid_weight) * focal_loss


# only for training
class HybridFocalLoss(nn.Module):
    def __init__(
        self,
        dim: str,
        hybrid_weight: float,
        tversky_fore_weight: float,
        tversky_fore_power: float,
        tversky_fp_weight: float,
        bce_fore_weight: float,
        bce_back_power: float,
    ):
        super().__init__()
        self.__dim = dim
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
            dim=self.__dim,
            hybrid_weight=self.hybrid_weight,
            tversky_fore_weight=self.tversky_fore_weight,
            tversky_fore_power=self.tversky_fore_power,
            tversky_fp_weight=self.tversky_fp_weight,
            bce_fore_weight=self.bce_fore_weight,
            bce_back_power=self.bce_back_power,
        )


def __get_surface_distances(
    preds: Union[Tensor, ndarray], labels: Union[Tensor, ndarray]
) -> float:
    if preds.shape != labels.shape:
        g.exit_app("__get_surface_distances(): preds.shape != labels.shape")

    if g.DEVICE != torch.device("cpu"):
        try:
            preds = preds.cpu()
            labels = labels.cpu()
        except AttributeError:
            pass

    if isinstance(labels, Tensor):
        preds = preds.numpy()
        labels = labels.numpy()

    preds = preds.squeeze()
    labels = labels.squeeze()

    if len(labels.shape) == 2:
        spacing_mm = g.NII_SPACING[:2]
    else:
        spacing_mm = g.NII_SPACING

    preds = np.where(preds >= 0.5, 1, 0)
    labels = np.where(labels >= 0.5, 1, 0)
    preds = preds.astype(np.bool)
    labels = labels.astype(np.bool)
    surface_distances = sd.compute_surface_distances(
        labels, preds, spacing_mm=spacing_mm
    )
    return surface_distances


def __test_dsc(preds: Union[Tensor, ndarray], labels: Union[Tensor, ndarray]) -> float:
    tp = (preds * labels).sum()
    dsc = (2 * tp + SMOOTH) / (preds.sum() + labels.sum() + SMOOTH)
    return dsc.item()


def __test_msd(preds: Union[Tensor, ndarray], labels: Union[Tensor, ndarray]) -> float:
    surface_distances = __get_surface_distances(preds, labels)
    msd = sd.compute_average_surface_distance(surface_distances)
    msd = (msd[0] + msd[1]) / 2
    if math.isnan(msd) or math.isinf(msd):
        return g.IMG_SIZE
    else:
        return msd


def __test_hd95(preds: Union[Tensor, ndarray], labels: Union[Tensor, ndarray]) -> float:
    surface_distances = __get_surface_distances(preds, labels)
    hd95 = sd.compute_robust_hausdorff(surface_distances, 95)
    if math.isnan(hd95) or math.isinf(hd95):
        return g.IMG_SIZE
    else:
        return hd95


def test_2d_score(
    preds: Union[Tensor, ndarray],
    labels: Union[Tensor, ndarray],
    score_type: str,
    binarize: bool = True,
):
    if binarize:
        preds = g.binarize_img(preds)

    slices_score_list = []

    depth = labels.size(0)

    for i in range(depth):
        if preds[i].sum() == 0 and labels[i].sum() == 0:
            score = "empty"
        elif preds[i].sum() == 0:
            score = "no.pred"
        elif labels[i].sum() == 0:
            score = "no.label"
        else:  # only test slices with preds and labels
            if score_type == "dsc":
                score = __test_dsc(preds=preds[i], labels=labels[i])
            elif score_type == "msd":
                score = __test_msd(preds=preds[i], labels=labels[i])
            elif score_type == "hd95":
                score = __test_hd95(preds=preds[i], labels=labels[i])

        slices_score_list.append(score)

    return slices_score_list


def test_3d_score(
    preds: Union[Tensor, ndarray],
    labels: Union[Tensor, ndarray],
    score_type: str,
    binarize: bool = True,
):
    if binarize:
        preds = g.binarize_img(preds)

    if preds.sum() == 0 and labels.sum() == 0:
        score = "empty"
    elif preds.sum() == 0:
        score = "no.pred"
    elif labels.sum() == 0:
        score = "no.label"
    else:  # only test slices with preds and labels
        if score_type == "dsc":
            score = __test_dsc(preds=preds, labels=labels)
        elif score_type == "msd":
            score = __test_msd(preds=preds, labels=labels)
        elif score_type == "hd95":
            score = __test_hd95(preds=preds, labels=labels)
    return score


# only for testing
class ScoreFunction(nn.Module):
    def __init__(
        self,
        score_type: str,
        dim: str,
    ):
        super().__init__()
        self.__score_type = score_type
        self.dim = dim

    def forward(self, preds: Tensor, labels: Tensor):
        if preds.shape != labels.shape:
            g.exit_app("ScoreFunction(): preds.shape != labels.shape")

        if self.dim == "2d":
            return test_2d_score(
                preds=preds,
                labels=labels,
                score_type=self.__score_type,
            )
        else:
            return test_3d_score(
                preds=preds,
                labels=labels,
                score_type=self.__score_type,
            )
