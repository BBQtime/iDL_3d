import torch
import torch.nn as nn
from custom import Dict
from custom import Global as g
from str_lib import StrLib as s
from torch import Tensor, tensor


# only for training
# dice loss: weight=1.0, delta=0.5
class UnifiedFocalLoss(nn.Module):
    def _split_channels(self, input_imgs: Tensor) -> Dict:
        # dimension: [batch, channel, depth, height, width]
        output_imgs = Dict()

        output_imgs[s.BACKGROUND] = input_imgs[:, 0, :, :, :]
        output_imgs[s.GTVT] = input_imgs[:, 1, :, :, :]
        output_imgs[s.GTVN] = input_imgs[:, 2, :, :, :]

        return output_imgs

    def __focal_loss(
        self, preds: dict, labels: dict, weight_map: Tensor = None
    ) -> Tensor:
        loss = Dict()

        # calculate loss through each channel
        for i in preds.keys():
            # clip values to prevent division by zero error
            preds[i] = torch.clip(preds[i], self.__epsilon, 1.0 - self.__epsilon)

            # cross entropy
            loss[i] = -labels[i] * torch.log(preds[i])

            # always suppress background
            # suppress foreground when symmetrical
            if i == s.BACKGROUND or not self.__asym:
                # suppression (larger gamma, more suppression)
                # loss always > 0, both before and after this step
                loss[i] = loss[i] * torch.pow(1 - preds[i], self.__gamma)

            # weight given to foreground (delta) amd background (1-delta)
            if i == s.BACKGROUND:
                loss[i] = loss[i] * (1 - self.__delta)
            else:
                loss[i] = loss[i] * self.__delta

            # add weight to loss
            if weight_map is not None and i == s.GTVT:
                loss[i] = loss[i] * weight_map

        loss = loss.to_list()
        loss = torch.stack(loss, dim=-1)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss

    def __focal_tversky_loss(
        self, preds: dict, labels: dict, weight_map: Tensor = None
    ) -> Tensor:
        axis = (1, 2, 3)

        tp = Dict()
        fn = Dict()
        fp = Dict()
        loss = Dict()

        # calculate loss through each channel
        for i in preds.keys():
            if weight_map is not None and i == s.GTVT:
                tp[i] = torch.sum(labels[i] * preds[i] * weight_map, dim=axis)
                fn[i] = torch.sum(labels[i] * (1 - preds[i]) * weight_map, dim=axis)
                fp[i] = torch.sum((1 - labels[i]) * preds[i] * weight_map, dim=axis)
            else:
                tp[i] = torch.sum(labels[i] * preds[i], dim=axis)
                fn[i] = torch.sum(labels[i] * (1 - preds[i]), dim=axis)
                fp[i] = torch.sum((1 - labels[i]) * preds[i], dim=axis)

            loss[i] = (tp[i] + self.__epsilon) / (
                tp[i]
                + self.__delta * fn[i]
                + (1 - self.__delta) * fp[i]
                + self.__epsilon
            )
            loss[i] = 1 - loss[i]

            # always enhance foreground, enhance background when symmetrical
            # larger gamma, more enhancement
            if i != s.BACKGROUND or not self.__asym:
                # clip values to prevent division by zero error
                loss[i] = torch.clip(loss[i], self.__epsilon)
                # loss always > 0, both before and after this step
                loss[i] = loss[i] * torch.pow(loss[i], -self.__gamma)

        loss = loss.to_list()
        loss = torch.stack(loss, dim=-1)
        loss = torch.mean(loss)

        return loss

    def __unified_focal_loss(
        self, preds: Tensor, labels: Tensor, weight_map: Tensor = None
    ) -> Tensor:
        if weight_map is not None:
            # weight_map only has one channel
            # weight_map: [b,c,d,h,w] -> [b,d,h,w],
            weight_map = weight_map[:, 0, :, :, :]

        if self.__weight > 0:
            ftl = self.__focal_tversky_loss(
                preds=preds, labels=labels, weight_map=weight_map
            )
        else:
            ftl = tensor(0, dtype=torch.float32)

        if self.__weight < 1:
            fl = self.__focal_loss(preds=preds, labels=labels, weight_map=weight_map)
        else:
            fl = tensor(0, dtype=torch.float32)

        return self.__weight * ftl + (1 - self.__weight) * fl

    def __init__(self, asym: bool, weight: float, delta: float, gamma: float):
        super().__init__()

        # asym: =true for imbalanced dataset,
        # in Focal Tversky loss, asym will enhance foreground
        # in Focal loss, asym will suppress background
        self.__asym = asym

        # weight: lambda parameter, controls balance between Focal Tversky loss and Focal loss
        self.__weight = weight

        # delta: controls weight given to each class
        # in Focal Tversky loss, delta controls weight given to FN (delta) and FP (1-delta)
        # in Focal loss, delta controls weight given to foreground (delta) amd background (1-delta)
        self.__delta = delta

        # gamma: focal parameter controls the degree of background suppression or foreground enhancement
        # in Focal Tversky loss, gamma controls the degree of foreground enhancement
        # in Focal loss, gamma controls the degree of background suppression
        self.__gamma = gamma

        self.__epsilon = torch.finfo(torch.float32).eps

    def forward(self, preds, labels, weight_map=None):
        preds = self._split_channels(preds)
        labels = self._split_channels(labels)
        return self.__unified_focal_loss(preds, labels, weight_map)
