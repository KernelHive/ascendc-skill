import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Basic Linear Attention mechanism.
    Uses feature maps to reduce attention computation from O(n^2*d) to O(n*d^2).
    Changes computation order: first compute K^T V, then multiply by Q.
    """

    def __init__(self, d_model, n_heads, feature_map='elu'):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            feature_map: Feature map type ('elu', 'relu', 'identity')
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.eps = 1e-6

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if feature_map == 'elu':
            self.feature_map_fn = lambda x: F.elu(x) + 1
        elif feature_map == 'relu':
            self.feature_map_fn = F.relu
        elif feature_map == 'identity':
            self.feature_map_fn = lambda x: x
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")

    def forward(self, x):
        """
        Forward pass for linear attention.
        Core idea: change computation order - compute K^T V first, then multiply by Q.
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply feature map (makes attention non-negative)
        Q = self.feature_map_fn(Q)
        K = self.feature_map_fn(K)

        # Linear attention computation
        # 1. Compute K^T V: [batch, heads, d_k, d_k]
        KV = torch.matmul(K.transpose(-2, -1), V)

        # 2. Compute normalization factor: [batch, heads, seq_len, 1]
        Z = 1 / (torch.einsum('bhnd,bhd->bhn', Q, K.sum(dim=2)).unsqueeze(-1) + self.eps)

        # 3. Compute output: Q(K^T V)
        output = torch.matmul(Q, KV) * Z

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, 'elu']
