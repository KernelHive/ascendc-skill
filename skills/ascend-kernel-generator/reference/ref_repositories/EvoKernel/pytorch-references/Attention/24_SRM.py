import torch
import torch.nn as nn


class Model(nn.Module):
    """SRM: A Style-based Recalibration Module for Convolutional Neural Networks.

    Extracts style information from each channel via style pooling (mean and
    standard deviation), then estimates per-channel recalibration weights
    through channel-independent style integration using a grouped 1D
    convolution followed by batch normalization and sigmoid gating.
    """

    def __init__(self, channel):
        super().__init__()
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, groups=channel,
                             bias=False)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.shape
        # style pooling
        mean = x.reshape(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.reshape(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat([mean, std], dim=-1)
        # style integration
        z = self.cfc(u)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels]
