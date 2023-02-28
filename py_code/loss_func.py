import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

EPSILON = torch.finfo(torch.float32).eps

# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        # keras channels_last(default): (b, d, h, w, c)
        # keras channels_first: (b, c, d, h, w)
        # return [1, 2, 3]
        # pytorch (b, c, d, h, w)
        return (2, 3, 4)
    # Two dimensional
    elif len(shape) == 4:
        # keras channels_last(default): (b, h, w, c)
        # keras channels_first: (b, c, h, w)
        # return [1, 2]
        # pytorch (b, c, h, w)
        return (2, 3)
    # Exception - Unknown
    else:
        raise ValueError("Metric: Shape of tensor is neither 2D or 3D.")


def split_input_data(input_imgs: Tensor, gtvt_only: bool):
    output_imgs = dict()

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
    gtvt_only: bool,
):
    def loss_function(input_pred, input_label):
        loss = dict()

        pred = split_input_data(input_pred, gtvt_only)
        label = split_input_data(input_label, gtvt_only)

        if gtvt_only:
            chan_list = ["back", "gtvt"]
        else:
            chan_list = ["back", "gtvt", "gtvn"]

        # calculate loss through each channel
        for i in chan_list:
            # clip values to prevent division by zero error
            pred[i] = torch.clip(pred[i], EPSILON, 1.0 - EPSILON)

            # cross entropy
            loss[i] = -label[i] * torch.log(pred[i])

            # always suppress background
            # suppress foreground when symmetrical
            if i == "back" or not asym:
                # suppression (larger gamma, more suppression)
                loss[i] = loss[i] * torch.pow(1 - pred[i], gamma)

            # weight given to foreground (delta) amd background (1-delta)
            if i == "back":
                loss[i] = loss[i] * (1 - delta)
            else:
                loss[i] = loss[i] * delta

        # average loss
        loss_list = []
        for i in chan_list:
            loss_list.append(loss[i])

        return torch.mean(torch.sum(torch.stack(loss_list, dim=-1), dim=-1))

    return loss_function


def focal_tversky_loss(
    asym: bool,  # =true for imbalanced dataset, only enhance foreground
    delta: float,  # weight given to false negative (delta) and false positive (1-delta)
    gamma: float,  # focal parameter controls the degree of foreground enhancement
    gtvt_only: bool,
):
    def loss_function(input_pred, input_label):
        tp = dict()
        fn = dict()
        fp = dict()
        loss = dict()

        pred = split_input_data(input_pred, gtvt_only)
        label = split_input_data(input_label, gtvt_only)

        if gtvt_only:
            chan_list = ["back", "gtvt"]
        else:
            chan_list = ["back", "gtvt", "gtvn"]

        # calculate loss through each channel
        axis = (1, 2, 3)
        for i in chan_list:
            # clip values to prevent division by zero error
            pred[i] = torch.clip(pred[i], EPSILON, 1.0 - EPSILON)

            tp[i] = torch.sum(label[i] * pred[i], dim=axis)
            fn[i] = torch.sum(label[i] * (1 - pred[i]), dim=axis)
            fp[i] = torch.sum((1 - label[i]) * pred[i], dim=axis)
            loss[i] = (tp[i] + EPSILON) / (
                tp[i] + delta * fn[i] + (1 - delta) * fp[i] + EPSILON
            )
            loss[i] = 1 - loss[i]

            # always enhance foreground
            # enhance background when symmetrical
            if i != "back" or not asym:
                # enhancement (larger gamma, more enhancement)
                loss[i] = loss[i] * torch.pow(loss[i], -gamma)

        # average loss
        loss_list = []
        for i in chan_list:
            loss_list.append(loss[i])

        return torch.mean(torch.stack(loss_list, dim=-1))

    return loss_function


def unified_focal_loss(
    asym: bool,  # asym=true for imbalanced dataset, only enhance foreground or suppress background
    weight: float,  # lambda parameter, controls weight given to Focal Tversky loss and Focal loss
    delta: float,  # controls weight given to each class
    gamma: float,  # focal parameter controls the degree of background suppression and foreground enhancement
    gtvt_only: bool,
):
    def loss_function(pred, label):

        ftl = focal_tversky_loss(
            asym=asym,
            delta=delta,
            gamma=gamma,
            gtvt_only=gtvt_only,
        )(pred, label)

        fl = focal_loss(
            asym=asym,
            delta=delta,
            gamma=gamma,
            gtvt_only=gtvt_only,
        )(pred, label)

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

    def forward(self, pred, label):
        if pred.shape != label.shape:
            raise ValueError("preds.shape != labels.shape")

        return self.__loss_func(pred, label)
