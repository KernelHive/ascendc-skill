import torch
import torch.nn as nn
from torch.nn import init


class Model(nn.Module):
    """
    SpatialGroupEnhance (SGE): Spatial Group-wise Enhance module.

    Groups channels and enhances spatial attention within each group by
    computing similarity between each spatial position and the global
    average-pooled feature, followed by normalization and sigmoid gating.
    """

    def __init__(self, groups=8):
        super(Model, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g, dim//g, h, w
        xn = x * self.avg_pool(x)  # bs*g, dim//g, h, w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g, 1, h, w
        t = xn.view(b * self.groups, -1)  # bs*g, h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g, h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g, h*w
        t = t.view(b, self.groups, h, w)  # bs, g, h, w

        t = t * self.weight + self.bias  # bs, g, h, w
        t = t.view(b * self.groups, 1, h, w)  # bs*g, 1, h, w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [8]
