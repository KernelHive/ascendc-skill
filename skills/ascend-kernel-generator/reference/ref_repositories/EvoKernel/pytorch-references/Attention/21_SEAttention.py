import torch
import torch.nn as nn


class Model(nn.Module):
    """Squeeze-and-Excitation (SE) Attention Module.

    Adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels using global average pooling
    followed by a two-layer fully-connected bottleneck.
    """

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


batch_size = 128
in_channels = 512
height = 7
width = 7
reduction = 16


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, reduction]
