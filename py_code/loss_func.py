import torch
import torch.nn as nn
import global_elems as g
from torch import Tensor
from torch import tensor
from custom import Dict

EPSILON = torch.finfo(torch.float32).eps


def split_channels(input_imgs: Tensor, gtvt_only: bool) -> dict:

    output_imgs = Dict()

    # dimension: [batch, channel, depth, height, width]
    output_imgs["gtvt"] = input_imgs[:, 1, :, :, :]
    if gtvt_only:
        output_imgs["back"] = input_imgs[:, 0, :, :, :] + input_imgs[:, 2, :, :, :]
    else:
        output_imgs["back"] = input_imgs[:, 0, :, :, :]
        output_imgs["gtvn"] = input_imgs[:, 2, :, :, :]

    return output_imgs


def focal_loss(
    asym: bool,  # asym=true for imbalanced dataset, only suppress background
    delta: float,  # weight given to foreground (delta) amd background (1-delta)
    gamma: float,  # focal parameter controls the degree of background suppression
):
    def loss_function(pred: dict, label: dict, weight_map: Tensor = None) -> Tensor:

        loss = Dict()

        # calculate loss through each channel
        for i in pred.keys():

            # clip values to prevent division by zero error
            pred[i] = torch.clip(pred[i], EPSILON, 1.0 - EPSILON)

            # cross entropy
            loss[i] = -label[i] * torch.log(pred[i])

            # always suppress background
            # suppress foreground when symmetrical
            if i == "back" or not asym:
                # suppression (larger gamma, more suppression)
                # loss always > 0, both before and after this step
                loss[i] = loss[i] * torch.pow(1 - pred[i], gamma)

            # weight given to foreground (delta) amd background (1-delta)
            if i == "back":
                loss[i] = loss[i] * (1 - delta)
            else:
                loss[i] = loss[i] * delta

            # add weight to loss
            if weight_map is not None:
                loss[i] = loss[i] * weight_map

        loss = g.dict_to_list(loss)
        loss = torch.stack(loss, dim=-1)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss

    return loss_function


def focal_tversky_loss(
    asym: bool,  # =true for imbalanced dataset, only enhance foreground
    delta: float,  # weight given to FN (delta) and FP (1-delta)
    gamma: float,  # focal parameter controls the degree of foreground enhancement
):
    axis = (1, 2, 3)

    def loss_function(pred: dict, label: dict, weight_map: Tensor = None) -> Tensor:

        tp = Dict()
        fn = Dict()
        fp = Dict()
        loss = Dict()

        # calculate loss through each channel
        for i in pred.keys():

            if weight_map is not None:
                tp[i] = torch.sum(label[i] * pred[i] * weight_map, dim=axis)
                fn[i] = torch.sum(label[i] * (1 - pred[i]) * weight_map, dim=axis)
                fp[i] = torch.sum((1 - label[i]) * pred[i] * weight_map, dim=axis)
            else:
                tp[i] = torch.sum(label[i] * pred[i], dim=axis)
                fn[i] = torch.sum(label[i] * (1 - pred[i]), dim=axis)
                fp[i] = torch.sum((1 - label[i]) * pred[i], dim=axis)

            loss[i] = (tp[i] + EPSILON) / (
                tp[i] + delta * fn[i] + (1 - delta) * fp[i] + EPSILON
            )
            loss[i] = 1 - loss[i]

            # always enhance foreground, enhance background when symmetrical
            # larger gamma, more enhancement
            if i != "back" or not asym:
                # clip values to prevent division by zero error
                loss[i] = torch.clip(loss[i], EPSILON)
                # loss always > 0, both before and after this step
                loss[i] = loss[i] * torch.pow(loss[i], -gamma)

        loss = g.dict_to_list(loss)
        loss = torch.stack(loss, dim=-1)
        loss = torch.mean(loss)

        return loss

    return loss_function


def unified_focal_loss(
    asym: bool,  # asym=true for imbalanced dataset, only enhance foreground or suppress background
    weight: float,  # lambda parameter, controls weight given to Focal Tversky loss and Focal loss
    delta: float,  # controls weight given to each class
    gamma: float,  # focal parameter controls the degree of background suppression and foreground enhancement
    gtvt_only: bool,
):
    def loss_function(pred: Tensor, label: Tensor, weight_map: Tensor = None) -> Tensor:

        pred = split_channels(pred, gtvt_only)
        label = split_channels(label, gtvt_only)

        if weight_map is not None:
            # weight_map: [b,c,d,h,w] -> [b,d,h,w]
            weight_map = weight_map[:, 0, :, :, :]

        if weight > 0:
            ftl = focal_tversky_loss(asym=asym, delta=delta, gamma=gamma)(
                pred=pred,
                label=label,
                weight_map=weight_map,
            )
        else:
            ftl = tensor(0, dtype=torch.float32)

        if weight < 1:
            fl = focal_loss(asym=asym, delta=delta, gamma=gamma)(
                pred=pred,
                label=label,
                weight_map=weight_map,
            )
        else:
            fl = tensor(0, dtype=torch.float32)

        return (weight * ftl) + (1 - weight) * fl

    return loss_function


# only for training
class UnifiedFocalLoss(nn.Module):
    def __init__(
        self,
        asym: bool,
        weight: float,
        delta: float,
        gamma: float,
        gtvt_only: bool,
    ):
        super().__init__()
        self.__loss_func = unified_focal_loss(
            asym=asym,
            weight=weight,
            delta=delta,
            gamma=gamma,
            gtvt_only=gtvt_only,
        )

    def forward(self, pred, label, weight_map):
        if pred.shape != label.shape:
            raise ValueError("preds.shape != labels.shape")

        return self.__loss_func(pred, label, weight_map)
