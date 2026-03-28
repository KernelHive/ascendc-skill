import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, d_k, dropout=0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        batch_size, n_heads, seq_len, d_k = Q.size()
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Parallelizes attention computation across h heads.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, value.size(1), self.n_heads, self.d_k).transpose(1, 2)

        attn_output = self.attention(Q, K, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output


class Model(MultiHeadAttention):
    """
    Self-Attention layer.
    Query, key, and value all come from the same input.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        """
        return super().forward(x, x, x)


# Configuration
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads]
