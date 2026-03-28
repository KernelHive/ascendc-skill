import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashAttentionV2(nn.Module):
    """
    Flash Attention v2 with improved parallelization,
    reduced shared memory usage, and optional attention biases.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class Model(nn.Module):
    """
    Benchmark wrapper for FlashAttentionV2.
    Implements Flash Attention v2 with improved parallelization strategy.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = FlashAttentionV2(
            d_model=d_model,
            n_heads=n_heads,
        )

    def forward(self, x):
        return self.attn(x)


# Configuration
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8


def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]


def get_init_inputs():
    return [d_model, n_heads]
