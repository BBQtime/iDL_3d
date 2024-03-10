import torch
import torch.nn as nn
from custom import GPU
from custom import Global as g
from torch import Tensor


class VGGBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, dropout: float = 0):
        super().__init__()

        # padding_mode: use "replicate" or "zeros"
        # "reflect" is not implemented by Pytorch yet
        # "circular" has bugs
        self.double_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(inplace=True),
        )
        if dropout > 0:
            self.double_conv = nn.Sequential(self.double_conv, nn.Dropout3d(dropout))

    def forward(self, input_data: Tensor) -> Tensor:
        return self.double_conv(input_data)


class UNetSlim(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        dataset_ver: str,
        edge_chan: int = 16,  # make it 64 for origin UNetSlim
        dropout: float = 0,
    ):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # vgg blocks
        self.vgg = nn.ModuleDict()
        for i in range(0, 4 + 1):
            # nn.ModuleDict module name must be "str"
            self.vgg[str(i)] = nn.ModuleDict()

        self.vgg["0"]["0"] = VGGBlock(in_chan, edge_chan)
        self.vgg["1"]["0"] = VGGBlock(edge_chan, edge_chan * 2)
        self.vgg["2"]["0"] = VGGBlock(edge_chan * 2, edge_chan * 4, dropout)
        self.vgg["3"]["0"] = VGGBlock(edge_chan * 4, edge_chan * 8, dropout)
        self.vgg["4"]["0"] = VGGBlock(edge_chan * 8, edge_chan * 16, dropout)

        self.vgg["3"]["1"] = VGGBlock(
            edge_chan * 8 + edge_chan * 16, edge_chan * 8, dropout
        )
        self.vgg["2"]["2"] = VGGBlock(
            edge_chan * 4 + edge_chan * 8, edge_chan * 4, dropout
        )
        self.vgg["1"]["3"] = VGGBlock(edge_chan * 2 + edge_chan * 4, edge_chan * 2)
        self.vgg["0"]["4"] = VGGBlock(edge_chan + edge_chan * 2, edge_chan)

        # upsample layers
        self.up = nn.ModuleDict()
        for i in range(1, 4 + 1):
            # nn.ModuleDict module name must be "str"
            self.up[str(i)] = nn.ModuleDict()

        kernel = 2

        self.up["4"]["0"] = nn.ConvTranspose3d(edge_chan * 16, edge_chan * 16, 2, 2)
        self.up["3"]["1"] = nn.ConvTranspose3d(edge_chan * 8, edge_chan * 8, 2, 2)
        self.up["2"]["2"] = nn.ConvTranspose3d(edge_chan * 4, edge_chan * 4, 2, 2)
        self.up["1"]["3"] = nn.ConvTranspose3d(
            edge_chan * 2, edge_chan * 2, kernel, kernel
        )

        # pooling layers
        self.pool = nn.ModuleDict()
        # nn.ModuleDict module name must be "str"
        self.pool["0"] = nn.MaxPool3d(kernel_size=kernel, stride=kernel)
        self.pool["1"] = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool["2"] = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool["3"] = nn.MaxPool3d(kernel_size=2, stride=2)

        # final layer
        self.final = nn.Conv3d(edge_chan, out_chan, kernel_size=1, stride=1)
        if out_chan == 1:
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        else:
            self.final = nn.Sequential(self.final, nn.Softmax())

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

    def forward(self, input_data: Tensor) -> Tensor:
        vgg = self.vgg
        pool = self.pool
        up = self.up

        # go down
        x00 = vgg["0"]["0"](input_data)
        x10 = pool["0"](x00)
        x10 = vgg["1"]["0"](x10)
        x20 = pool["1"](x10)
        x20 = vgg["2"]["0"](x20)
        x30 = pool["2"](x20)
        x30 = vgg["3"]["0"](x30)
        x40 = pool["3"](x30)
        x40 = vgg["4"]["0"](x40)

        # row 3
        x31 = up["4"]["0"](x40)
        x31 = vgg["3"]["1"](torch.cat([x30, x31], 1))

        # row 2
        x22 = up["3"]["1"](x31)
        x22 = vgg["2"]["2"](torch.cat([x20, x22], 1))

        # row 1
        x13 = up["2"]["2"](x22)
        x13 = vgg["1"]["3"](torch.cat([x10, x13], 1))

        # row 0
        x04 = up["1"]["3"](x13)
        x04 = vgg["0"]["4"](torch.cat([x00, x04], 1))

        output = self.final(x04)
        return output


# for testing
if 0:
    img_size = 256
    img_depth = 256
    batch_size = 1
    in_chan = 4
    out_chan = 3

    cnn = UNetSlim(in_chan, out_chan)
    if GPU.used_count() > 1:
        cnn = nn.DataParallel(cnn)
    cnn = cnn.to(g.DEVICE)

    input_data = torch.rand(
        batch_size,
        in_chan,
        img_depth,
        img_size,
        img_size,
    ).to(g.DEVICE)

    print(input_data.shape)
    output_data = cnn.forward(input_data)
    print(output_data.shape)
