import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Local (Sliding Window) Attention mechanism.
    Each position attends only to positions within a fixed-size window around it.
    Complexity: O(n * window_size * d)
    """

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.0)

    def create_local_mask(self, seq_len, window_size, device):
        """
        Create a local window mask.

        Args:
            seq_len: sequence length
            window_size: window size
            device: device

        Returns:
            mask: [seq_len, seq_len] mask matrix
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input tensor [batch_size, seq_len, d_model]

        Returns:
            output: output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        mask = self.create_local_mask(seq_len, self.window_size, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
window_size = 32


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, window_size]
