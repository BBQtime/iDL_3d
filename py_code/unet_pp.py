import global_elems as g
import torch
import torch.nn as nn
from torch import Tensor


class UNetPP(nn.Module):
    def __freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def __unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def freeze_top(self):
        # freeze vgg blocks
        for j in range(0, 5):  # [0, 4]
            for i in range(5 - j):  # [0, 4/3/2/1/0]
                # skip vgg["4"]["0"]
                if i == 4 and j == 0:
                    pass
                else:
                    self.__freeze_layer(self.vgg[i][j])

        # freeze up sample layer ["4"]["0"]
        for j in range(4):  # [0, 3]
            for i in range(4 - j):  # [0, 3/2/1/0]
                # skip up["4"]["0"]
                if (i + 1) == 4 and j == 0:
                    pass
                else:
                    self.__freeze_layer(self.up[i + 1][j])

        # freeze pooling layer ["3"]
        for i in range(4):
            if i == 3:
                pass
            else:
                self.__freeze_layer(self.pool[i])

    def unfreeze_top(self):
        # freeze vgg blocks
        for j in range(5):  # [0, 4]
            for i in range(5 - j):  # [0, 4/3/2/1/0]
                self.__unfreeze_layer(self.vgg[i][j])

        # freeze up sample layer ["4"]["0"]
        for j in range(4):  # [0, 3]
            for i in range(4 - j):  # [0, 3/2/1/0]
                self.__unfreeze_layer(self.up[i + 1][j])

        # freeze pooling layer ["3"]
        for i in range(4):
            self.__unfreeze_layer(self.pool[i])

    def __init__(
        self, in_channels: int = 4, out_channels: int = 3, dropout: float = 0.0
    ):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # vgg blocks
        self.vgg = nn.ModuleDict()
        for i in range(0, 4 + 1):
            # nn.ModuleDict module name must be "str"
            self.vgg[str(i)] = nn.ModuleDict()

        self.vgg["0"]["0"] = VGGBlock(in_channels, 32)
        self.vgg["1"]["0"] = VGGBlock(32, 64)
        self.vgg["2"]["0"] = VGGBlock(64, 128)
        self.vgg["3"]["0"] = VGGBlock(128, 256, dropout)
        self.vgg["4"]["0"] = VGGBlock(256, 512, dropout)

        self.vgg["0"]["1"] = VGGBlock(32 + 64, 32)
        self.vgg["1"]["1"] = VGGBlock(64 + 128, 64)
        self.vgg["2"]["1"] = VGGBlock(128 + 256, 128, dropout)
        self.vgg["3"]["1"] = VGGBlock(256 + 512, 256, dropout)

        self.vgg["0"]["2"] = VGGBlock(32 * 2 + 64, 32)
        self.vgg["1"]["2"] = VGGBlock(64 * 2 + 128, 64)
        self.vgg["2"]["2"] = VGGBlock(128 * 2 + 256, 128, dropout)

        self.vgg["0"]["3"] = VGGBlock(32 * 3 + 64, 32)
        self.vgg["1"]["3"] = VGGBlock(64 * 3 + 128, 64)

        self.vgg["0"]["4"] = VGGBlock(32 * 4 + 64, 32)

        # upsample layers
        self.up = nn.ModuleDict()
        for i in range(1, 4 + 1):
            # nn.ModuleDict module name must be "str"
            self.up[str(i)] = nn.ModuleDict()

        self.up["1"]["0"] = nn.ConvTranspose3d(64, 64, 2, 2)
        self.up["2"]["0"] = nn.ConvTranspose3d(128, 128, 2, 2)
        self.up["3"]["0"] = nn.ConvTranspose3d(256, 256, 2, 2)
        self.up["4"]["0"] = nn.ConvTranspose3d(512, 512, 2, 2)

        self.up["1"]["1"] = nn.ConvTranspose3d(64, 64, 2, 2)
        self.up["2"]["1"] = nn.ConvTranspose3d(128, 128, 2, 2)
        self.up["3"]["1"] = nn.ConvTranspose3d(256, 256, 2, 2)

        self.up["1"]["2"] = nn.ConvTranspose3d(64, 64, 2, 2)
        self.up["2"]["2"] = nn.ConvTranspose3d(128, 128, 2, 2)

        self.up["1"]["3"] = nn.ConvTranspose3d(64, 64, 2, 2)

        # pooling layers
        self.pool = nn.ModuleDict()
        for i in range(0, 3 + 1):
            # nn.ModuleDict module name must be "str"
            self.pool[str(i)] = nn.MaxPool3d(kernel_size=2, stride=2)

        # final layers
        self.final = nn.Conv3d(32, out_channels, kernel_size=1, stride=1)
        if out_channels == 1:
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        else:
            self.final = nn.Sequential(self.final, nn.Softmax())

    def forward(self, input_data: Tensor) -> Tensor:
        vgg = self.vgg
        pool = self.pool
        up = self.up

        x00 = vgg["0"]["0"](input_data)

        x10 = vgg["1"]["0"](pool["0"](x00))
        x01 = vgg["0"]["1"](torch.cat([x00, up["1"]["0"](x10)], 1))

        x20 = vgg["2"]["0"](pool["1"](x10))
        x11 = vgg["1"]["1"](torch.cat([x10, up["2"]["0"](x20)], 1))
        x02 = vgg["0"]["2"](torch.cat([x00, x01, up["1"]["1"](x11)], 1))

        x30 = vgg["3"]["0"](pool["2"](x20))
        x21 = vgg["2"]["1"](torch.cat([x20, up["3"]["0"](x30)], 1))
        x12 = vgg["1"]["2"](torch.cat([x10, x11, up["2"]["1"](x21)], 1))
        x03 = vgg["0"]["3"](torch.cat([x00, x01, x02, up["1"]["2"](x12)], 1))

        x40 = vgg["4"]["0"](pool["3"](x30))
        x31 = vgg["3"]["1"](torch.cat([x30, up["4"]["0"](x40)], 1))
        x22 = vgg["2"]["2"](torch.cat([x20, x21, up["3"]["1"](x31)], 1))
        x13 = vgg["1"]["3"](torch.cat([x10, x11, x12, up["2"]["2"](x22)], 1))
        x04 = vgg["0"]["4"](torch.cat([x00, x01, x02, x03, up["1"]["3"](x13)], 1))

        # deep supervision:
        if 0:
            output1 = self.final(x01)
            output2 = self.final(x02)
            output3 = self.final(x03)
            output4 = self.final(x04)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x04)
            return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        # padding_mode: use "replicate" or "zeros"
        # "reflect" is not implemented by Pytorch yet
        # "circular" is said to have bugs
        self.double_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        if dropout > 0:
            self.double_conv = nn.Sequential(self.double_conv, nn.Dropout3d(dropout))

    def forward(self, input_data: Tensor) -> Tensor:
        return self.double_conv(input_data)


# # for testing
# if 1:
#     batch_size = 1
#     in_channels = 4
#     out_channels = 3
#     g.clear_gpu_cache()
#     cnn = UNetPP(in_channels, out_channels)
#     # cnn = nn.ConvTranspose3d(in_channels, out_channels, 2, 2)
#     if g.used_gpu_count() > 1:
#         cnn = nn.DataParallel(cnn)
#     cnn = cnn.to(g.DEVICE)
#     input_data = torch.rand(batch_size, in_channels, 96, 128, 128).to(g.DEVICE)
#     print(input_data.shape)
#     output_data = cnn.forward(input_data)
#     # output_data = cnn(input_data)
#     print(output_data.shape)
