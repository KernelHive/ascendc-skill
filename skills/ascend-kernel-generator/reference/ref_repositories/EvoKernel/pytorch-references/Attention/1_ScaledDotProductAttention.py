import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Scaled Dot-Product Attention.
    Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, d_k, dropout=0.0):
        """
        Args:
            d_k: Key dimension.
            dropout: Dropout probability (default 0.0 for inference).
        """
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        """
        Forward pass.

        Args:
            Q: Query tensor [batch_size, n_heads, seq_len, d_k]
            K: Key tensor [batch_size, n_heads, seq_len, d_k]
            V: Value tensor [batch_size, n_heads, seq_len, d_v]

        Returns:
            output: Attention output [batch_size, n_heads, seq_len, d_v]
        """
        batch_size, n_heads, seq_len, d_k = Q.size()

        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Compute output: attention weights x V
        output = torch.matmul(attn_weights, V)

        return output


# Configuration
batch_size = 2
n_heads = 8
seq_len = 128
d_k = 64


def get_inputs():
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)
    return [Q, K, V]


def get_init_inputs():
    return [d_k]
