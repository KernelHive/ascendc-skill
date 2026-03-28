import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MoE UnPermute: restore original token order and weighted sum.

    After expert processing, tokens are in expert-first order. This operator
    restores them to the original token order using the inverse permutation,
    then applies the routing weights and sums across the topk experts.

    Input:     (total_expanded, hidden_size), bfloat16 — expert outputs
    topk_vals: (num_tokens, topk), bfloat16 — routing weights
    inv_perm:  (num_tokens * topk,), int64 — inverse permutation indices
    Output:    (num_tokens, hidden_size), bfloat16
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, expert_output, topk_vals, inv_perm):
        M, topk = topk_vals.shape
        K = expert_output.shape[1]

        # Reorder using inverse permutation
        reordered = expert_output[inv_perm]  # (M * topk, K)

        # Reshape to (M, topk, K)
        reordered = reordered.view(M, topk, K)

        # Apply routing weights and sum across experts
        weighted = reordered * topk_vals.unsqueeze(-1)  # (M, topk, K)
        out = weighted.sum(dim=1)  # (M, K)

        return out


# Configuration — MoE dimensions
num_tokens = 512
hidden_size = 4096
topk = 8
num_groups = 64
block_m = 128

# Total rows after permute (padded)
tokens_per_expert = ((num_tokens * topk // num_groups + block_m - 1) // block_m) * block_m
total_expanded = num_groups * tokens_per_expert


def get_inputs():
    expert_output = torch.randn(total_expanded, hidden_size, dtype=torch.bfloat16)
    topk_vals = torch.randn(num_tokens, topk, dtype=torch.bfloat16).softmax(dim=-1)
    # inv_perm maps each of num_tokens*topk slots to a position in expert_output
    inv_perm = torch.randperm(total_expanded)[:num_tokens * topk]
    return [expert_output, topk_vals, inv_perm]


def get_init_inputs():
    return [topk]
