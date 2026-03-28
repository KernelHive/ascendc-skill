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


class Model(nn.Module):
    """
    Multi-Head Attention mechanism.
    Parallelizes attention computation across h heads, each learning
    different representation subspaces.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        """
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability (default 0.0 for inference).
            bias: Whether to use bias in linear projections.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # Attention computation
        self.attention = ScaledDotProductAttention(self.d_k, dropout)

        # Final dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Forward pass.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]

        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, value.size(1), self.n_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output = self.attention(Q, K, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output


# Configuration
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8


def get_inputs():
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    return [query, key, value]


def get_init_inputs():
    return [d_model, n_heads]
