import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Block Sparse Attention mechanism.
    Divides the sequence into fixed-size blocks and computes attention only within each block.
    Complexity: O(n * block_size * d)
    """

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input tensor [batch_size, seq_len, d_model]

        Returns:
            output: output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        assert seq_len % self.block_size == 0, \
            f"Sequence length {seq_len} must be divisible by block size {self.block_size}"
        n_blocks = seq_len // self.block_size

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Reshape into blocks: [batch_size, n_heads, n_blocks, block_size, d_k]
        Q = Q.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k)
        K = K.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k)
        V = V.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k)

        # Intra-block attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # Reshape back to original shape
        output = output.view(batch_size, self.n_heads, seq_len, self.d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
block_size = 32


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, block_size]
