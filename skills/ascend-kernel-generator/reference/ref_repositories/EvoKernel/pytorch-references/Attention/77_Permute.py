import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MoE Permute: token-first to expert-first reordering.

    Given hidden_states and topk_ids from the Router, reorders tokens so that
    all tokens assigned to the same expert are contiguous. Pads each expert's
    block to a multiple of block_m for efficient grouped GEMM.

    Input:     (num_tokens, hidden_size), bfloat16
    topk_ids:  (num_tokens, topk), int32 — expert assignment per token
    Output:    (num_groups * tokens_per_expert_padded, hidden_size), bfloat16
    m_indices: (num_groups * tokens_per_expert_padded,), int32 — expert index per row
    inv_perm:  (num_tokens * topk,), int64 — inverse permutation for unpermute
    """

    def __init__(self, num_groups, topk, block_m):
        super().__init__()
        self.num_groups = num_groups
        self.topk = topk
        self.block_m = block_m

    def forward(self, hidden_states, topk_ids):
        M, K = hidden_states.shape
        num_tokens = M * self.topk

        # Count tokens per expert
        flat_ids = topk_ids.reshape(-1)  # (M * topk,)
        counts = torch.zeros(self.num_groups, dtype=torch.int64, device=hidden_states.device)
        for i in range(self.num_groups):
            counts[i] = (flat_ids == i).sum()

        # Pad each expert's count to a multiple of block_m
        padded_counts = ((counts + self.block_m - 1) // self.block_m) * self.block_m

        # Build sorted token IDs: group by expert
        sorted_token_ids = []
        m_indices_list = []

        for expert_id in range(self.num_groups):
            # Find all token slots assigned to this expert
            mask = (flat_ids == expert_id)
            token_slots = torch.where(mask)[0]  # indices into flat_ids

            # Pad with the last valid index (clamped)
            pad_count = padded_counts[expert_id].item() - len(token_slots)
            if len(token_slots) > 0:
                padded = torch.cat([
                    token_slots,
                    token_slots[-1:].expand(int(pad_count))
                ]) if pad_count > 0 else token_slots
            else:
                # No tokens for this expert — fill with dummy
                padded = torch.zeros(int(padded_counts[expert_id].item()),
                                     dtype=torch.long, device=hidden_states.device)

            sorted_token_ids.append(padded)
            m_indices_list.append(
                torch.full((int(padded_counts[expert_id].item()),), expert_id,
                           dtype=torch.int32, device=hidden_states.device))

        sorted_token_ids = torch.cat(sorted_token_ids)
        m_indices = torch.cat(m_indices_list)

        # Clamp to valid range
        sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)

        # Compute inverse permutation (for unpermute)
        inv_perm = torch.argsort(sorted_token_ids)[:num_tokens]

        # Gather: each slot maps to a token via token_slot // topk
        source_token_idx = sorted_token_ids // self.topk
        output = hidden_states[source_token_idx]

        return output, m_indices, inv_perm


# Configuration — MoE dimensions
num_tokens = 512
hidden_size = 4096
num_groups = 64     # number of experts
topk = 8
block_m = 128       # alignment block size


def get_inputs():
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, num_groups, (num_tokens, topk), dtype=torch.int32)
    return [hidden_states, topk_ids]


def get_init_inputs():
    return [num_groups, topk, block_m]
