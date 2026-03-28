import torch
import torch.nn as nn

# Source:
# - Repo: MarcoDotIO/mhc-deepseek-implementation | Path: src/mhc/stream_ops.py | URL: https://github.com/MarcoDotIO/mhc-deepseek-implementation/blob/main/src/mhc/stream_ops.py



class Model(nn.Module):
    """Weighted sum over residual streams: (B,T,n,C) + (B,T,n) -> (B,T,C)."""

    def __init__(self):
        super().__init__()

    def forward(self, x_stream: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.dtype != x_stream.dtype:
            weights = weights.to(dtype=x_stream.dtype)
        return torch.einsum("btn,btnc->btc", weights, x_stream)


# Configuration
batch_size = 8
seq_len = 128
num_streams = 4
dim = 256


def get_inputs():
    x_stream = torch.randn(batch_size, seq_len, num_streams, dim)
    weights = torch.randn(batch_size, seq_len, num_streams)
    return [x_stream, weights]


def get_init_inputs():
    return []
