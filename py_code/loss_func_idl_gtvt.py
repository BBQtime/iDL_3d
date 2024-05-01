from custom_dict import Dict
from loss_func import UnifiedFocalLoss
from torch import Tensor


class UnifiedFocalLossIDLGTVt(UnifiedFocalLoss):
    def _split_channels(self, input_imgs: Tensor) -> Dict:
        # dimension: [batch, channel, depth, height, width]
        output_imgs = Dict()

        # preds have 3 channels, background = background + gtvn
        if input_imgs.shape[1] == 3:
            output_imgs["background"] = (
                input_imgs[:, 0, :, :, :] + input_imgs[:, 2, :, :, :]
            )

        # labels have 2 channels, background = background
        elif input_imgs.shape[1] == 2:
            output_imgs["background"] = input_imgs[:, 0, :, :, :]

        output_imgs["gtvt"] = input_imgs[:, 1, :, :, :]

        return output_imgs
