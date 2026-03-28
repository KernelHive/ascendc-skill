import torch
import torch.nn as nn

# Source:
# - Repo: MarcoDotIO/mhc-deepseek-implementation | Path: src/mhc/stream_ops.py | URL: https://github.com/MarcoDotIO/mhc-deepseek-implementation/blob/main/src/mhc/stream_ops.py



class Model(nn.Module):
    """Write layer output back into streams: (B,T,C) + (B,T,n) -> (B,T,n,C)."""

    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
        if h_post.dtype != y.dtype:
            h_post = h_post.to(dtype=y.dtype)
        return h_post.unsqueeze(-1) * y.unsqueeze(-2)


# Configuration
batch_size = 8
seq_len = 128
num_streams = 4
dim = 256


def get_inputs():
    y = torch.randn(batch_size, seq_len, dim)
    h_post = torch.randn(batch_size, seq_len, num_streams)
    return [y, h_post]


def get_init_inputs():
    return []
