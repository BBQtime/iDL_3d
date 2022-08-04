import global_elems as g
import torch
import torch.nn as nn
from nested_dict import NestedDict
from torch import Tensor


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        if dropout > 0:
            self.double_conv = nn.Sequential(self.double_conv, nn.Dropout(dropout))

    def forward(self, input_data: Tensor) -> Tensor:
        return self.double_conv(input_data)


class UNetPP2D(nn.Module):
    def freeze_top(self):
        # freeze vgg blocks
        for j in range(5):  # [0, 4]
            for i in range(5 - j):  # [0, 4/3/2/1/0]
                # skip vgg["4"]["0"]
                if i == 4 and j == 0:
                    pass
                else:
                    self.__freeze_layer(self.vgg[str(i)][str(j)])

        # freeze up sample layer ["4"]["0"]
        for j in range(4):  # [0, 3]
            for i in range(4 - j):  # [0, 3/2/1/0]
                # skip up["4"]["0"]
                if (i + 1) == 4 and j == 0:
                    pass
                else:
                    self.__freeze_layer(self.up[str(i + 1)][str(j)])

        # freeze pooling layer ["3"]
        for i in range(4):
            if i == 3:
                pass
            else:
                self.__freeze_layer(self.pool[str(i)])

    def unfreeze_top(self):
        # freeze vgg blocks
        for j in range(5):  # [0, 4]
            for i in range(5 - j):  # [0, 4/3/2/1/0]
                self.__unfreeze_layer(self.vgg[str(i)][str(j)])

        # freeze up sample layer ["4"]["0"]
        for j in range(4):  # [0, 3]
            for i in range(4 - j):  # [0, 3/2/1/0]
                self.__unfreeze_layer(self.up[str(i + 1)][str(j)])

        # freeze pooling layer ["3"]
        for i in range(4):
            self.__unfreeze_layer(self.pool[str(i)])

    # def freeze_encoder(self):
    #     for i in range(5):
    #         self.__freeze_layer(self.vgg[str(i)]["0"])
    #     for i in range(4):
    #         self.__freeze_layer(self.pool[str(i)])

    # def unfreeze_encoder(self):
    #     for i in range(5):
    #         self.__unfreeze_layer(self.vgg[str(i)]["0"])
    #     for i in range(4):
    #         self.__unfreeze_layer(self.pool[str(i)])

    def __freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def __unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def __init__(
        self, in_channels: int = 4, out_channels: int = 1, dropout: float = 0.0
    ):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # vgg blocks
        self.vgg = nn.ModuleDict()
        for i in range(5):
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
        for i in range(4):
            self.up[str(i + 1)] = nn.ModuleDict()

        self.up["1"]["0"] = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up["2"]["0"] = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up["3"]["0"] = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up["4"]["0"] = nn.ConvTranspose2d(512, 512, 2, 2)

        self.up["1"]["1"] = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up["2"]["1"] = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up["3"]["1"] = nn.ConvTranspose2d(256, 256, 2, 2)

        self.up["1"]["2"] = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up["2"]["2"] = nn.ConvTranspose2d(128, 128, 2, 2)

        self.up["1"]["3"] = nn.ConvTranspose2d(64, 64, 2, 2)

        # pooling layers
        self.pool = nn.ModuleDict()
        for i in range(4):
            self.pool[str(i)] = nn.MaxPool2d(kernel_size=2, stride=2)

        # final layers
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        if out_channels == 1:
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        else:
            self.final = nn.Sequential(self.final, nn.Softmax2d())

    def forward(self, input_data: Tensor) -> Tensor:
        x = NestedDict()
        vgg = self.vgg
        pool = self.pool
        up = self.up

        x[0][0] = vgg["0"]["0"](input_data)

        x[1][0] = vgg["1"]["0"](pool["0"](x[0][0]))
        x[0][1] = vgg["0"]["1"](torch.cat([x[0][0], up["1"]["0"](x[1][0])], 1))

        x[2][0] = vgg["2"]["0"](pool["1"](x[1][0]))
        x[1][1] = vgg["1"]["1"](torch.cat([x[1][0], up["2"]["0"](x[2][0])], 1))
        x[0][2] = vgg["0"]["2"](torch.cat([x[0][0], x[0][1], up["1"]["1"](x[1][1])], 1))

        x[3][0] = vgg["3"]["0"](pool["2"](x[2][0]))
        x[2][1] = vgg["2"]["1"](torch.cat([x[2][0], up["3"]["0"](x[3][0])], 1))
        x[1][2] = vgg["1"]["2"](torch.cat([x[1][0], x[1][1], up["2"]["1"](x[2][1])], 1))
        x[0][3] = vgg["0"]["3"](
            torch.cat([x[0][0], x[0][1], x[0][2], up["1"]["2"](x[1][2])], 1)
        )

        x[4][0] = vgg["4"]["0"](pool["3"](x[3][0]))
        x[3][1] = vgg["3"]["1"](torch.cat([x[3][0], up["4"]["0"](x[4][0])], 1))
        x[2][2] = vgg["2"]["2"](torch.cat([x[2][0], x[2][1], up["3"]["1"](x[3][1])], 1))
        x[1][3] = vgg["1"]["3"](
            torch.cat([x[1][0], x[1][1], x[1][2], up["2"]["2"](x[2][2])], 1)
        )
        x[0][4] = vgg["0"]["4"](
            torch.cat(
                [
                    x[0][0],
                    x[0][1],
                    x[0][2],
                    x[0][3],
                    up["1"]["3"](x[1][3]),
                ],
                1,
            )
        )

        # deep supervision:
        if 0:
            output1 = self.final(x[0][1])
            output2 = self.final(x[0][2])
            output3 = self.final(x[0][3])
            output4 = self.final(x[0][4])
            return [output1, output2, output3, output4]

        else:
            output = self.final(x[0][4])
            return output


# for testing
# if 0:
#     batch_size = 8
#     in_channels = 4
#     out_channels = 2
#     g.clear_gpu_cache()
#     cnn = UNetPP2D(in_channels, out_channels).to(g.DEVICE)
#     input_data = torch.rand(batch_size, in_channels, 512, 512).to(g.DEVICE)
#     print(input_data.shape)
#     # g.show_img(input_data.cpu())
#     output_data = cnn.forward(input_data)
#     print(output_data.shape)
#     # g.show_img(output_data.cpu())
