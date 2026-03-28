import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Flash Attention forward computation with GQA support, causal masking,
    sliding window attention, and softcap stabilization.

    Operates directly on Q, K, V tensors without linear projections.
    Supports Grouped Query Attention (GQA) where nheads_q > nheads_kv.

    Input Q:  (batch, seqlen_q, nheads, headdim), bfloat16
    Input K:  (batch, seqlen_k, nheads_kv, headdim), bfloat16
    Input V:  (batch, seqlen_k, nheads_kv, headdim), bfloat16
    Output:   (batch, seqlen_q, nheads, headdim), bfloat16
    """

    def __init__(self, nheads, nheads_kv, headdim, causal=True,
                 window_size=(-1, -1), softcap=0.0):
        super().__init__()
        self.nheads = nheads
        self.nheads_kv = nheads_kv
        self.headdim = headdim
        self.causal = causal
        self.window_size = window_size
        self.softcap = softcap
        self.scale = 1.0 / math.sqrt(headdim)
        assert nheads % nheads_kv == 0
        self.n_groups = nheads // nheads_kv

    def forward(self, q, k, v):
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, nheads_kv, _ = k.shape

        # Transpose to (batch, heads, seqlen, headdim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: repeat K, V heads to match Q head count
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Compute attention scores: (B, nheads, seqlen_q, seqlen_k)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale

        # Apply softcap for numerical stability
        if self.softcap > 0.0:
            scores = self.softcap * torch.tanh(scores / self.softcap)

        # Build mask for causal and/or sliding window
        row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        # diff: relative position of col to row (adjusted for seqlen difference)
        diff = col_idx - (row_idx + seqlen_k - seqlen_q)

        mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)

        # Causal mask: cannot attend to future positions
        if self.causal:
            mask |= (diff > 0)

        # Sliding window mask
        if self.window_size[0] >= 0:
            mask |= (diff < -self.window_size[0])
        if self.window_size[1] >= 0:
            mask |= (diff > self.window_size[1])

        if mask.any():
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax in float32 for numerical stability
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        out = torch.matmul(attn_weights, v.float())

        # Transpose back to (batch, seqlen_q, nheads, headdim)
        out = out.transpose(1, 2).to(q.dtype)

        return out


# Configuration — realistic LLM dimensions
batch_size = 2
seqlen_q = 512
seqlen_k = 512
nheads = 16
nheads_kv = 8
headdim = 128
causal = True
window_size_left = -1
window_size_right = -1
softcap = 0.0


def get_inputs():
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, dtype=torch.bfloat16)
    return [q, k, v]


def get_init_inputs():
    return [nheads, nheads_kv, headdim, causal,
            (window_size_left, window_size_right), softcap]
