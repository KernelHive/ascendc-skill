import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Linformer Attention mechanism.
    Uses low-rank projection to reduce key/value sequence length from n to k.
    Achieves linear complexity by projecting K and V to lower dimensions.
    Paper: Linformer: Self-Attention with Linear Complexity
    """

    def __init__(self, d_model, n_heads, seq_len, k):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            seq_len: Maximum sequence length
            k: Projection dimension (k << seq_len)
        """
        super().__init__()
        assert d_model % n_heads == 0
        assert k < seq_len, "Projection dimension k must be less than sequence length"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Low-rank projection matrices
        self.E = nn.Parameter(torch.randn(seq_len, k))  # Projection for K
        self.F = nn.Parameter(torch.randn(seq_len, k))  # Projection for V

        # Initialize projection matrices
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

    def forward(self, x):
        """
        Forward pass with low-rank projected attention.
        Projects K and V from sequence length n to k dimensions before attention computation.
        """
        batch_size, seq_len, _ = x.size()

        # Handle variable-length sequences
        if seq_len != self.seq_len:
            E = F.adaptive_avg_pool1d(
                self.E.unsqueeze(0).transpose(1, 2),
                seq_len
            ).transpose(1, 2).squeeze(0)
            F_proj = F.adaptive_avg_pool1d(
                self.F.unsqueeze(0).transpose(1, 2),
                seq_len
            ).transpose(1, 2).squeeze(0)
        else:
            E = self.E
            F_proj = self.F

        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Low-rank projection: reduce K and V sequence dimension from n to k
        # K: [batch, heads, seq_len, d_k] -> [batch, heads, k, d_k]
        K = torch.matmul(E.T.unsqueeze(0).unsqueeze(0), K)
        # V: [batch, heads, seq_len, d_k] -> [batch, heads, k, d_k]
        V = torch.matmul(F_proj.T.unsqueeze(0).unsqueeze(0), V)

        # Standard attention with reduced K and V dimensions
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
k = 32


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, seq_len, k]
