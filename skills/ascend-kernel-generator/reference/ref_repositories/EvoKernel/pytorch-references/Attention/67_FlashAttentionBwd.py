import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Flash Attention backward computation. Computes gradients dq, dk, dv
    for the multi-head attention operation with GQA support and causal masking.

    Input dout: (batch, seqlen_q, nheads, headdim), bfloat16
    Input Q:    (batch, seqlen_q, nheads, headdim), bfloat16
    Input K:    (batch, seqlen_k, nheads_kv, headdim), bfloat16
    Input V:    (batch, seqlen_k, nheads_kv, headdim), bfloat16
    Output dq:  (batch, seqlen_q, nheads, headdim), bfloat16
    Output dk:  (batch, seqlen_k, nheads_kv, headdim), bfloat16
    Output dv:  (batch, seqlen_k, nheads_kv, headdim), bfloat16
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

    def _attention_forward(self, q, k, v):
        """Forward pass used to build the computation graph for autograd."""
        batch, nheads, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape

        # GQA: repeat K, V heads
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.softcap > 0.0:
            scores = self.softcap * torch.tanh(scores / self.softcap)

        # Build attention mask
        row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        diff = col_idx - (row_idx + seqlen_k - seqlen_q)

        mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
        if self.causal:
            mask |= (diff > 0)
        if self.window_size[0] >= 0:
            mask |= (diff < -self.window_size[0])
        if self.window_size[1] >= 0:
            mask |= (diff > self.window_size[1])

        if mask.any():
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out

    def forward(self, dout, q, k, v):
        with torch.enable_grad():
            # Work in float32 for numerical stability
            q_f = q.float().detach().requires_grad_(True)
            k_f = k.float().detach().requires_grad_(True)
            v_f = v.float().detach().requires_grad_(True)

            # Transpose to (batch, heads, seqlen, headdim) for attention
            q_t = q_f.transpose(1, 2)
            k_t = k_f.transpose(1, 2)
            v_t = v_f.transpose(1, 2)

            out = self._attention_forward(q_t, k_t, v_t)
            # Transpose back to (batch, seqlen, heads, headdim)
            out = out.transpose(1, 2)

            out.backward(dout.float())

        dq = q_f.grad.to(dout.dtype)
        dk = k_f.grad.to(dout.dtype)
        dv = v_f.grad.to(dout.dtype)
        return dq, dk, dv


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
    dout = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=torch.bfloat16)
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, dtype=torch.bfloat16)
    return [dout, q, k, v]


def get_init_inputs():
    return [nheads, nheads_kv, headdim, causal,
            (window_size_left, window_size_right), softcap]
