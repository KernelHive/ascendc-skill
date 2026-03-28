import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Grouped Query Attention (GQA).
    Divides query heads into G groups, each group sharing one key-value pair.
    A compromise between MHA and MQA.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            max_seq_len: int = 2048,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key-value heads (must divide n_heads)
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Q uses all heads; K and V use fewer heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # 1. Compute Q (all heads)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Compute K and V (fewer heads)
        K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 3. Repeat K and V to match Q head count
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # 4. Standard attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # 5. Merge multi-head
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


# Configuration variables
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
n_kv_heads = 2
dropout = 0.0
max_seq_len = 2048


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, n_kv_heads]
