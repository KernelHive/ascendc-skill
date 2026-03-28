import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    BigBird Attention mechanism.
    Combines random attention, local sliding window attention, and global attention
    on the first and last tokens.
    """

    def __init__(self, d_model, n_heads, window_size, num_random_blocks):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.num_random_blocks = num_random_blocks
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.0)

    def create_bigbird_mask(self, seq_len, window_size, num_random_blocks, device):
        """
        Create a BigBird hybrid mask.
        Includes: local window + global tokens (first/last) + random attention.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)

        # 1. Local window attention
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

        # 2. Global attention (first and last tokens)
        mask[0, :] = 1   # First token attends to all
        mask[-1, :] = 1  # Last token attends to all
        mask[:, 0] = 1   # All attend to first token
        mask[:, -1] = 1  # All attend to last token

        # 3. Random attention
        for i in range(1, seq_len - 1):  # Exclude first and last
            random_indices = torch.randperm(seq_len)[:num_random_blocks]
            mask[i, random_indices] = 1

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

        mask = self.create_bigbird_mask(
            seq_len, self.window_size, self.num_random_blocks, x.device
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
num_random_blocks = 3


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, window_size, num_random_blocks]
