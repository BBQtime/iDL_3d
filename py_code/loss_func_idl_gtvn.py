from custom import Dict
from loss_func import UnifiedFocalLoss
from str_lib import BACKGROUND, GTVN
from torch import Tensor


class UnifiedFocalLossIDLGTVn(UnifiedFocalLoss):
    def _split_channels(self, input_imgs: Tensor) -> Dict:
        # dimension: [batch, channel, depth, height, width]
        output_imgs = Dict()

        output_imgs[BACKGROUND] = input_imgs[:, 0, :, :, :]
        output_imgs[GTVN] = input_imgs[:, 1, :, :, :]

        return output_imgs
