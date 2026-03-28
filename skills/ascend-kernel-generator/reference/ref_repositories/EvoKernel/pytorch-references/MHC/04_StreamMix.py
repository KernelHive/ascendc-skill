import torch
import torch.nn as nn

# Source:
# - Repo: MarcoDotIO/mhc-deepseek-implementation | Path: src/mhc/stream_ops.py | URL: https://github.com/MarcoDotIO/mhc-deepseek-implementation/blob/main/src/mhc/stream_ops.py



class Model(nn.Module):
    """Mix residual streams with H_res: (B,T,n,n) @ (B,T,n,C) -> (B,T,n,C)."""

    def __init__(self):
        super().__init__()

    def forward(self, x_stream: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        if h_res.dtype != x_stream.dtype:
            h_res = h_res.to(dtype=x_stream.dtype)
        return torch.einsum("btij,btjc->btic", h_res, x_stream)


# Configuration
batch_size = 8
seq_len = 128
num_streams = 4
dim = 256


def get_inputs():
    x_stream = torch.randn(batch_size, seq_len, num_streams, dim)
    h_res = torch.randn(batch_size, seq_len, num_streams, num_streams)
    return [x_stream, h_res]


def get_init_inputs():
    return []
