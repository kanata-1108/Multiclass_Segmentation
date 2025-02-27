import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.convblock(x)

        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconvblock = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.upconvblock(x)

        return out

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # down
        self.CB1 = ConvBlock(in_channels, 64)
        self.CB2 = ConvBlock(64, 128)
        self.CB3 = ConvBlock(128, 256)
        self.CB4 = ConvBlock(256, 512)
        self.CB5 = ConvBlock(512, 1024)

        # up
        self.UCB1 = UpConvBlock(1024, 512)
        self.CB6 = ConvBlock(1024, 512)
        self.UCB2 = UpConvBlock(512, 256)
        self.CB7 = ConvBlock(512, 256)
        self.UCB3 = UpConvBlock(256, 128)
        self.CB8 = ConvBlock(256, 128)
        self.UCB4 = UpConvBlock(128, 64)
        self.CB9 = ConvBlock(128, 64)

        # other parts
        self.dropout = nn.Dropout(p = 0.4)
        self.conv = nn.Conv2d(64, out_channels, kernel_size = 1, padding = "same")
        self.pool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        x1 = self.CB1(x)
        x2 = self.pool(x1)
        x2 = self.dropout(x2)

        x2 = self.CB2(x2)
        x3 = self.pool(x2)

        x3 = self.CB3(x3)
        x4 = self.pool(x3)
        x4 = self.dropout(x4)

        x4 = self.CB4(x4)
        x5 = self.pool(x4)

        x5 = self.CB5(x5)
        x5 = self.dropout(x5)

        z1 = self.UCB1(x5)
        z1 = torch.cat((x4, z1), dim = 1)
        z1 = self.CB6(z1)

        z2 = self.UCB2(z1)
        z2 = torch.cat((x3, z2), dim = 1)
        z2 = self.CB7(z2)

        z3 = self.UCB3(z2)
        z3 = torch.cat((x2, z3), dim = 1)
        z3 = self.CB8(z3)

        z4 = self.UCB4(z3)
        z4 = torch.cat((x1, z4), dim = 1)
        z4 = self.CB9(z4)

        z = self.conv(z4)

        return z