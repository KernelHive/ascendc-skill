import torch
from torch import nn
import math


class Model(nn.Module):
    """
    Global Filter module that applies frequency-domain filtering using FFT,
    element-wise multiplication with learnable complex weights, and inverse FFT.
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)
        return x


batch_size = 32
dim = 512
h = 7
w = 4

def get_inputs():
    return [torch.randn(batch_size, 49, dim)]

def get_init_inputs():
    return [dim, h, w]
