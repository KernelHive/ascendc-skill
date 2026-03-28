import torch
import torch.nn as nn
import torch.nn.functional as F


class CoTAttention(nn.Module):
    """Contextual Transformer Attention (CoT Attention).

    Uses contextual information from keys to guide dynamic attention matrix
    generation, combining static context (from convolution) with dynamic
    context (from attention) for enhanced feature representation.
    """

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h*w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class Model(nn.Module):
    """Benchmark wrapper for CoTAttention."""

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.cot_attention = CoTAttention(dim=dim, kernel_size=kernel_size)

    def forward(self, x):
        return self.cot_attention(x)


batch_size = 128
in_channels = 512
height = 7
width = 7
dim = 512
kernel_size = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [dim, kernel_size]
