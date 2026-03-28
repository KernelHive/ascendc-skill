import torch
import torch.nn as nn
import math


class FlashAttention(nn.Module):
    """
    Flash Attention via tiled computation and online softmax.
    Uses block-wise Q/K/V processing to reduce HBM access.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size_q: int = 64,
        block_size_kv: int = 64,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv

        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _flash_attention_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention forward pass with tiled computation and online softmax.

        Args:
            Q: [batch_size, n_heads, seq_len_q, d_k]
            K: [batch_size, n_heads, seq_len_kv, d_k]
            V: [batch_size, n_heads, seq_len_kv, d_k]

        Returns:
            output: [batch_size, n_heads, seq_len_q, d_k]
        """
        batch_size, n_heads, seq_len_q, d_k = Q.shape
        _, _, seq_len_kv, _ = K.shape

        Br = min(self.block_size_q, seq_len_q)
        Bc = min(self.block_size_kv, seq_len_kv)

        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, n_heads, seq_len_q), device=Q.device)

        # Outer loop: iterate over Q blocks
        for i in range(0, seq_len_q, Br):
            i_end = min(i + Br, seq_len_q)
            Qi = Q[:, :, i:i_end, :]

            Oi = torch.zeros_like(Qi)
            Li = torch.zeros((batch_size, n_heads, i_end - i), device=Q.device)
            Mi = torch.full((batch_size, n_heads, i_end - i), float('-inf'), device=Q.device)

            # Inner loop: iterate over K/V blocks
            for j in range(0, seq_len_kv, Bc):
                j_end = min(j + Bc, seq_len_kv)

                Kj = K[:, :, j:j_end, :]
                Vj = V[:, :, j:j_end, :]

                # Attention scores
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * self.scale

                # Online softmax update
                Mi_new = torch.max(Mi, Sij.max(dim=-1)[0])

                correction = torch.exp(Mi - Mi_new)

                Pij = torch.exp(Sij - Mi_new.unsqueeze(-1))
                Li_new = correction * Li + Pij.sum(dim=-1)

                Oi = correction.unsqueeze(-1) * Oi + torch.matmul(Pij, Vj)

                Mi = Mi_new
                Li = Li_new

            # Normalize output block
            Oi = Oi / Li.unsqueeze(-1).clamp(min=1e-6)

            O[:, :, i:i_end, :] = Oi
            L[:, :, i:i_end] = Li

        return O

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

        output = self._flash_attention_forward(Q, K, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class Model(nn.Module):
    """
    Benchmark wrapper for FlashAttention.
    Implements Flash Attention with tiled computation and online softmax
    for memory-efficient attention.
    """

    def __init__(self, d_model, n_heads, block_size_q, block_size_kv):
        super().__init__()
        self.attn = FlashAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
        )

    def forward(self, x):
        return self.attn(x)


# Configuration
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
block_size_q = 64
block_size_kv = 64


def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]


def get_init_inputs():
    return [d_model, n_heads, block_size_q, block_size_kv]
