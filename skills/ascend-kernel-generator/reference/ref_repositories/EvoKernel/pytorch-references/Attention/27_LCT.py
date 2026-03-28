import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear Context Transform (LCT) Block.

    Divides channels into groups and normalizes globally aggregated context
    features within each group to reduce disturbance from irrelevant channels.
    A learned linear transform of the normalized context features models
    global context for each channel independently.
    """

    def __init__(self, channels, groups, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.avgpool(x).view(batch_size, self.groups, -1)
        mean = y.mean(dim=-1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        y_norm = self.w.reshape(1, -1, 1, 1) * y_norm + self.b.reshape(1, -1, 1, 1)
        y_norm = self.sigmoid(y_norm)
        return x * y_norm.expand_as(x)


batch_size = 128
in_channels = 512
height = 7
width = 7
groups = 32
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, groups, eps]
