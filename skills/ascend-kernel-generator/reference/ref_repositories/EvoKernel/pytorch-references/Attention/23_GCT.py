import torch
import torch.nn as nn


class Model(nn.Module):
    """Gaussian Context Transformer (GCT).

    Achieves contextual feature excitation using a Gaussian function that
    satisfies the presupposed relationship between channel context and
    feature importance.
    """

    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        return x * y_transform.expand_as(x)


batch_size = 128
in_channels = 512
height = 7
width = 7
c = 2
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, c, eps]
