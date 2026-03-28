import torch
import torch.nn as nn

# Source:
# - Repo: MarcoDotIO/mhc-deepseek-implementation | Path: src/mhc/stream_ops.py | URL: https://github.com/MarcoDotIO/mhc-deepseek-implementation/blob/main/src/mhc/stream_ops.py



class Model(nn.Module):
    """mHC residual update: H_res x + H_post^T y."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_stream: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if h_res.dtype != x_stream.dtype:
            h_res = h_res.to(dtype=x_stream.dtype)
        if h_post.dtype != y.dtype:
            h_post = h_post.to(dtype=y.dtype)

        resmix = torch.einsum("btij,btjc->btic", h_res, x_stream)
        addy = h_post.unsqueeze(-1) * y.unsqueeze(-2)
        return resmix + addy


# Configuration
batch_size = 8
seq_len = 128
num_streams = 4
dim = 256


def get_inputs():
    x_stream = torch.randn(batch_size, seq_len, num_streams, dim)
    h_post = torch.randn(batch_size, seq_len, num_streams)
    h_res = torch.randn(batch_size, seq_len, num_streams, num_streams)
    y = torch.randn(batch_size, seq_len, dim)
    return [x_stream, h_post, h_res, y]


def get_init_inputs():
    return []
