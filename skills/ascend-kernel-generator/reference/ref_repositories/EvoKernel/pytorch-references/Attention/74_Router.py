import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MoE Router: expert selection via Softmax + TopK.

    Computes softmax over gate logits then selects the top-k experts
    per token with their corresponding routing weights.

    Input:  (num_tokens, num_experts), bfloat16  — gate logits
    Output: topk_ids  (num_tokens, topk), int64
            topk_vals (num_tokens, topk), bfloat16 — routing weights
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, gate_logits):
        # Softmax in float32 for numerical stability
        probs = torch.softmax(gate_logits.float(), dim=-1)
        topk_vals, topk_ids = torch.topk(probs, k=self.topk, dim=-1)
        return topk_ids, topk_vals.to(gate_logits.dtype)


# Configuration — MoE dimensions
num_tokens = 2048
num_experts = 64
topk = 8


def get_inputs():
    gate_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)
    return [gate_logits]


def get_init_inputs():
    return [topk]
