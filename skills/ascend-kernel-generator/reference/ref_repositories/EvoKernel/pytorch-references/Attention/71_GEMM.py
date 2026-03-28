import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Standard GEMM (General Matrix Multiplication) for LLM linear layers.

    Computes Y = X @ W^T where X is the activation and W is the weight matrix.
    This is the core computation of all Linear/fully-connected layers in LLMs.

    Input:  (M, K), bfloat16  — activation tensor
    Weight: (N, K), bfloat16  — weight matrix (stored transposed)
    Output: (M, N), bfloat16
    """

    def __init__(self, K, N):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, K, dtype=torch.bfloat16))

    def forward(self, x):
        # Y = X @ W^T, equivalent to nn.Linear without bias
        return torch.matmul(x, self.weight.t())


# Configuration — realistic LLM dimensions
M = 2048       # num_tokens (batch_size * seq_len)
K = 4096       # input feature dimension (hidden_size)
N = 4096       # output feature dimension


def get_inputs():
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [K, N]
