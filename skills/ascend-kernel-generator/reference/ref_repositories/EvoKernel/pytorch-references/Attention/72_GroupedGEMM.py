import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Grouped GEMM for MoE (Mixture of Experts) FFN layers.

    Each row of the input is assigned to a group (expert) via m_indices.
    Row i is multiplied by rhs[m_indices[i]], producing output row i.
    This batches many independent small GEMMs into one kernel launch.

    The actual operator uses FP8 inputs with per-block scaling factors.
    This reference uses bfloat16 for numerical correctness on CPU.

    Input lhs:       (m_sum, K), bfloat16
    Input rhs:       (num_groups, N, K), bfloat16
    Input m_indices: (m_sum,), int32  — group assignment per row
    Output:          (m_sum, N), bfloat16
    """

    def __init__(self, num_groups, N, K):
        super().__init__()
        self.num_groups = num_groups
        self.N = N
        self.K = K
        # Expert weight matrices: each expert has a (N, K) weight
        self.rhs = nn.Parameter(torch.randn(num_groups, N, K, dtype=torch.bfloat16))

    def forward(self, lhs, m_indices):
        m_sum = lhs.shape[0]
        out = torch.empty(m_sum, self.N, dtype=lhs.dtype, device=lhs.device)
        for i in range(m_sum):
            idx = m_indices[i].item()
            # out[i] = lhs[i] @ rhs[idx]^T
            out[i] = torch.matmul(lhs[i].unsqueeze(0), self.rhs[idx].t()).squeeze(0)
        return out


# Configuration — MoE dimensions
num_tokens = 512
topk = 8
m_sum = num_tokens * topk   # total rows after token expansion
K = 4096                     # input feature dimension
N = 1024                     # output feature dimension (FFN intermediate / num_experts)
num_groups = 64              # number of experts


def get_inputs():
    lhs = torch.randn(m_sum, K, dtype=torch.bfloat16)
    m_indices = torch.randint(0, num_groups, (m_sum,), dtype=torch.int32)
    return [lhs, m_indices]


def get_init_inputs():
    return [num_groups, N, K]
