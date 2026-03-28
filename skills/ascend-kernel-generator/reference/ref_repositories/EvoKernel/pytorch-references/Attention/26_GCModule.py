import torch
import torch.nn as nn


class Model(nn.Module):
    """GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond (GC Module).

    Combines context modeling (aggregating features of all positions into a
    global context) with a bottleneck feature transform and additive fusion
    to capture long-range dependencies efficiently.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.LayerNorm([channel // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1)
        )

    def context_modeling(self, x):
        b, c, h, w = x.shape
        input_x = x
        input_x = input_x.reshape(b, c, h * w)
        context = self.conv(x)
        context = context.reshape(b, 1, h * w).transpose(1, 2)
        out = torch.matmul(input_x, context)
        out = out.reshape(b, c, 1, 1)
        return out

    def forward(self, x):
        context = self.context_modeling(x)
        y = self.transform(context)
        return x + y


batch_size = 128
in_channels = 512
height = 7
width = 7
reduction = 16


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, reduction]
