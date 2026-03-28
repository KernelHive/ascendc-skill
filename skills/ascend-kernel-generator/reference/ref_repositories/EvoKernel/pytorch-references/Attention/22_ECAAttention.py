import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks.

    Proposes a local cross-channel interaction strategy without dimensionality
    reduction, efficiently implemented via a 1D convolution whose kernel size
    is adaptively determined by the number of channels.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


batch_size = 128
in_channels = 512
height = 7
width = 7
gamma = 2
b = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, gamma, b]
