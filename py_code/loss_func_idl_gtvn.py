from loss_func import UnifiedFocalLoss
from torch import Tensor
from custom import Dict


class UnifiedFocalLossIDLGTVn(UnifiedFocalLoss):
    def _split_channels(self, input_imgs: Tensor) -> Dict:

        # dimension: [batch, channel, depth, height, width]
        output_imgs = Dict()

        output_imgs["back"] = input_imgs[:, 0, :, :, :]
        output_imgs["gtvn"] = input_imgs[:, 1, :, :, :]

        return output_imgs
