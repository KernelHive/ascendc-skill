import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Implementation of Double Attention (A2-Net).

    Applies a two-step attention mechanism:
    1. Feature gating: gathers global descriptors via attention maps.
    2. Feature distribution: distributes global descriptors to each location via attention vectors.

    Reference: A2-Nets: Double Attention Networks (NeurIPS 2018)
    """

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        """
        Args:
            in_channels: Number of input channels.
            c_m: Number of feature channels for attention maps (keys).
            c_n: Number of feature channels for attention vectors (values).
            reconstruct: If True, project output back to in_channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b, c_m, h, w
        B = self.convB(x)  # b, c_n, h, w
        V = self.convV(x)  # b, c_n, h, w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=-1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=-1)
        # step 1: feature gating
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b, c_m, c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b, c_m, h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b, c_m, h, w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


batch_size = 128
in_channels = 512
height = 7
width = 7
c_m = 128
c_n = 128
reconstruct = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512, 128, 128, True]
