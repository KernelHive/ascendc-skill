import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Longformer Attention mechanism.
    Combines local sliding window attention with global attention on selected tokens.
    """

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.global_attention_indices = [0, 511]
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Additional parameters for global attention
        self.W_q_global = nn.Linear(d_model, d_model)
        self.W_k_global = nn.Linear(d_model, d_model)
        self.W_v_global = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.0)

    def create_longformer_mask(self, seq_len, window_size, global_indices, device):
        """
        Create a Longformer hybrid mask combining local window and global attention.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            # Local window
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

            # Global attention
            if i in global_indices:
                mask[i, :] = 1  # Global position attends to all positions
                mask[:, i] = 1  # All positions attend to global position

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

        mask = self.create_longformer_mask(
            seq_len, self.window_size, self.global_attention_indices, x.device
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

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
