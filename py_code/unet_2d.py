import torch
import torch.nn as nn
import global_elems as g
from torch import Tensor
from nested_dict import NestedDict


class UNet2D(nn.Module):
    def __crop_tensor(self, origin_tensor: Tensor, target_tensor: Tensor):
        origin_size = origin_tensor.size()[2]
        target_size = target_tensor.size()[2]
        if origin_size == target_size:
            return origin_tensor
        else:
            delta = int((origin_size - target_size) / 2)
            return origin_tensor[
                :, :, delta : origin_size - delta, delta : origin_size - delta
            ]

    def __concat_tensor(self, encoder_data: Tensor, decoder_data: Tensor):
        encoder_data = self.__crop_tensor(
            origin_tensor=encoder_data, target_tensor=decoder_data
        )
        concat_data = torch.cat([decoder_data, encoder_data], dim=1)
        return concat_data

    # architecture: MaxPool(optional) + Conv*2
    # returns the data of current depth,
    # that is able to concatenate with the decoder path
    def __encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        first_block: bool = False,
        dropout: float = 0,
    ) -> nn.Module:
        double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            self.__act_func,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            self.__act_func,
        )
        if dropout > 0:
            double_conv = nn.Sequential(double_conv, nn.Dropout(dropout))

        if first_block:
            return double_conv
        else:
            return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), double_conv)

    # architecture: Conv*2(optional) + UpSample
    # or: Conv*2(optional) + Conv + Sigmoid/Softmax
    # __decoder_block() returns the data given to the above depth,
    # to concatenate with data from the encoder path on the left
    def __decoder_block(
        self,
        in_channels: int,
        out_channels: int,
        first_block: bool = False,
        final_block: bool = False,
        dropout: float = 0,
    ) -> nn.Module:
        up_sample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        if first_block:
            return up_sample

        double_conv = nn.Sequential(
            nn.Conv2d(
                # in_channels * 2 because of concatenation
                in_channels=in_channels * 2,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(in_channels),
            self.__act_func,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(in_channels),
            self.__act_func,
        )
        if dropout > 0:
            double_conv = nn.Sequential(double_conv, nn.Dropout(dropout))

        if final_block:
            double_conv = nn.Sequential(
                double_conv,
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                ),
            )
            if out_channels == 1:
                return nn.Sequential(
                    double_conv,
                    nn.Sigmoid(),
                )
            else:
                return nn.Sequential(
                    double_conv,
                    nn.Softmax2d(),
                )
        else:
            return nn.Sequential(double_conv, up_sample)

    # Pytorch: forward function must return the output value
    def forward(self, input_data: Tensor) -> Tensor:
        # encoder (contracting path)
        encoder_data = NestedDict()
        encoder_data[0] = self.__encoder_blocks["0"](input_data)
        for i in range(5):  # i = [0:4]
            encoder_data[i + 1] = self.__encoder_blocks[str(i + 1)](encoder_data[i])

        # decoder (expansion path)
        # decoder_block_0 is only a up sample layer (without conv)
        output_data = self.__decoder_blocks["0"](encoder_data[5])
        for i in range(5):  # i = [0:4]
            output_data = self.__concat_tensor(encoder_data[4 - i], output_data)
            output_data = self.__decoder_blocks[str(i + 1)](output_data)

        return output_data

    # in_channels=4 (ct/pt/mr1/mr2)
    def __init__(
        self, in_channels: int = 4, out_channels: int = 1, dropout: float = 0.0
    ):
        super().__init__()

        # nn.ELU() IS NOT BETTER THAN nn.ELU()
        self.__act_func = nn.LeakyReLU(inplace=True)

        # encoder blocks has to be nn.ModuleDict() instead of a normal dict,
        # otherwise the optimizer can not find the nn parameters
        self.__encoder_blocks = nn.ModuleDict()
        # currently, ModuleDict only supports string keys
        self.__encoder_blocks["0"] = self.__encoder_block(
            in_channels, 64, first_block=True
        )
        self.__encoder_blocks["1"] = self.__encoder_block(64, 128)
        self.__encoder_blocks["2"] = self.__encoder_block(128, 256)
        self.__encoder_blocks["3"] = self.__encoder_block(256, 512)
        self.__encoder_blocks["4"] = self.__encoder_block(512, 1024, dropout=dropout)
        # one more depth than standard UNet
        self.__encoder_blocks["5"] = self.__encoder_block(1024, 2048, dropout=dropout)

        # decoder blocks
        self.__decoder_blocks = nn.ModuleDict()
        self.__decoder_blocks["0"] = self.__decoder_block(
            2048, 1024, first_block=True, dropout=dropout
        )
        self.__decoder_blocks["1"] = self.__decoder_block(1024, 512, dropout=dropout)
        self.__decoder_blocks["2"] = self.__decoder_block(512, 256)
        self.__decoder_blocks["3"] = self.__decoder_block(256, 128)
        self.__decoder_blocks["4"] = self.__decoder_block(128, 64)
        self.__decoder_blocks["5"] = self.__decoder_block(
            64, out_channels, final_block=True
        )

    def freeze_top(self):
        # encoder, only update block_5
        for i in range(5):  # [0, 4]
            self.__freeze_layers(self.__encoder_blocks[str(i)])
        # encoder, only update block_0
        for i in range(5):  # [1, 5]
            self.__freeze_layers(self.__decoder_blocks[str(i + 1)])

    def unfreeze_top(self):
        for i in range(5):  # [0, 4]
            self.__unfreeze_layers(self.__encoder_blocks[str(i)])
        for i in range(5):  # [1, 5]
            self.__unfreeze_layers(self.__decoder_blocks[str(i + 1)])

    # def freeze_encoder(self):
    #     for layer in self.__encoder_blocks.values():
    #         self.__freeze_layers(layer)

    # def unfreeze_encoder(self):
    #     for layer in self.__encoder_blocks.values():
    #         self.__unfreeze_layers(layer)

    # freeze/unfreeze param in layers
    def __freeze_layers(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def __unfreeze_layers(self, layer):
        for param in layer.parameters():
            param.requires_grad = True


# for testing
# if 0:
#     batch_size = 8
#     in_channels = 4
#     out_channels = 2
#     g.clear_gpu_cache()
#     cnn = UNet2D(in_channels, out_channels).to(g.DEVICE)
#     input_data = torch.rand(batch_size, in_channels, 512, 512).to(g.DEVICE)
#     print(input_data.shape)
#     # g.show_img(input_data.cpu())
#     output_data = cnn.forward(input_data)
#     print(output_data.shape)
#     # g.show_img(output_data.cpu())
