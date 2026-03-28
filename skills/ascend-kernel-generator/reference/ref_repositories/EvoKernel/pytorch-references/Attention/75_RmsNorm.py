import torch
import torch.nn as nn


class Model(nn.Module):
    """
    RMS Normalization (Root Mean Square Layer Normalization).

    Normalizes inputs by dividing by the root mean square, then scales
    by a learnable weight parameter. Used in modern LLMs (LLaMA, etc.)
    as a simpler and faster alternative to LayerNorm.

    Input:  (num_tokens, hidden_size), bfloat16
    Output: (num_tokens, hidden_size), bfloat16
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(x.dtype)


# Configuration — realistic LLM dimensions
num_tokens = 2048
hidden_size = 4096
eps = 1e-6


def get_inputs():
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [hidden_size, eps]
