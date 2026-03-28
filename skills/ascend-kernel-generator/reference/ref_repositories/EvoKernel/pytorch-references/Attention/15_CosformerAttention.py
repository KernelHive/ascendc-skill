import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Cosformer Attention mechanism.
    Uses cosine-based re-weighting to achieve linear attention.
    Applies ReLU feature map and exponential positional decay for re-weighting.
    Paper: COSFORMER: Rethinking Softmax in Attention
    """

    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
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

    def forward(self, x):
        """
        Forward pass with cosine re-weighted linear attention.
        Applies ReLU feature map and exponential positional decay before
        computing linear attention via the K^T V trick.
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply cosine kernel (ReLU feature map)
        Q = F.relu(Q)
        K = F.relu(K)

        # Add position-based re-weighting with exponential decay
        position_indices = torch.arange(seq_len, device=x.device).float()
        position_indices = position_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Apply exponential decay
        decay = torch.exp(-position_indices / seq_len)
        Q = Q * decay
        K = K * decay

        # Linear attention computation
        KV = torch.matmul(K.transpose(-2, -1), V)
        Z = 1.0 / (torch.einsum('bhnd,bhd->bhn', Q, K.sum(dim=2)).unsqueeze(-1) + self.eps)
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
    return [d_model, n_heads]
