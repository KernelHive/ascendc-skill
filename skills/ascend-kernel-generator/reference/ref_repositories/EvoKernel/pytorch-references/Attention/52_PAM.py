import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Position Attention Module (PAM) from Dual Attention Network.

    Computes position-wise self-attention over the spatial dimensions
    of a feature map, learning to aggregate features across all positions
    weighted by their pairwise affinities.

    Reference: Dual Attention Network for Scene Segmentation (CVPR 2019)
    """

    def __init__(self, dim):
        """
        Args:
            dim: Number of input channels.
        """
        super().__init__()
        self.b = nn.Conv2d(dim, dim, 1)
        self.c = nn.Conv2d(dim, dim, 1)
        self.d = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, c, h, w = x.shape
        B = self.b(x).flatten(2).transpose(1, 2)
        C = self.c(x).flatten(2)
        D = self.d(x).flatten(2).transpose(1, 2)
        attn = (B @ C).softmax(dim=-1)
        y = (attn @ D).transpose(1, 2).reshape(n, c, h, w)
        out = self.alpha * y + x
        return out


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512]
