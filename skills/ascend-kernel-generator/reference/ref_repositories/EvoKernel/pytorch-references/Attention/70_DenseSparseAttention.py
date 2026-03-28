import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Dense-Sparse Attention (DSA) using FlashMLA-style sparse selection.

    Each query position selects topk KV tokens via pre-computed indices,
    rather than attending to the full KV sequence. The KV cache uses the
    same compressed single-head format as MLA (headdim_qk=576).

    Optionally supports attention sink weights for sink tokens.

    Input Q:       (batch, seqlen_q, nheads, headdim_qk), bfloat16
    Input kv_cache:(num_blocks, page_block_size, 1, headdim_qk), bfloat16
    Input indices: (batch, seqlen_q, topk), int32 — selected KV token positions
    Output:        (batch, seqlen_q, nheads, headdim_v), bfloat16
    """

    def __init__(self, nheads, headdim_qk, headdim_v, page_block_size, topk):
        super().__init__()
        self.nheads = nheads
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.page_block_size = page_block_size
        self.topk = topk
        self.scale = 1.0 / math.sqrt(headdim_qk)

    def forward(self, q, kv_cache, indices):
        batch, seqlen_q, nheads, headdim_qk = q.shape
        num_blocks, pbs, _, _ = kv_cache.shape

        # Flatten KV cache: (num_blocks * page_block_size, 1, headdim_qk)
        flat_kv = kv_cache.reshape(num_blocks * pbs, 1, self.headdim_qk)

        outputs = []
        for b in range(batch):
            out_seq = []
            for s in range(seqlen_q):
                # Gather topk KV entries for this query position
                idx = indices[b, s]  # (topk,)
                # Clamp to valid range
                idx = idx.clamp(0, num_blocks * pbs - 1).long()

                # Selected KV: (topk, 1, headdim_qk)
                selected_kv = flat_kv[idx]

                # K for scoring: (1, 1, topk, headdim_qk)
                k = selected_kv[:, :, :self.headdim_qk].unsqueeze(0).transpose(1, 2).float()
                # V for output: (1, 1, topk, headdim_v)
                v = selected_kv[:, :, :self.headdim_v].unsqueeze(0).transpose(1, 2).float()

                # Query for this position: (1, nheads, 1, headdim_qk)
                q_pos = q[b, s].unsqueeze(0).unsqueeze(2).float()  # (1, nheads, 1, headdim_qk)

                # Attention scores: (1, nheads, 1, topk)
                # k broadcasts from (1, 1, topk, headdim_qk) to all nheads
                scores = torch.matmul(q_pos, k.transpose(-2, -1)) * self.scale

                attn_weights = torch.softmax(scores, dim=-1)

                # Output: (1, nheads, 1, headdim_v)
                out_pos = torch.matmul(attn_weights, v)
                out_pos = out_pos.squeeze(2)  # (1, nheads, headdim_v)
                out_seq.append(out_pos)

            # Stack sequence positions: (1, seqlen_q, nheads, headdim_v)
            out_b = torch.stack(out_seq, dim=1).to(q.dtype)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)


# Configuration — DeepSeek V3 sparse attention dimensions
batch_size = 4
seqlen_q = 1           # Decoding mode
nheads = 16
headdim_qk = 576       # Compressed QK dimension
headdim_v = 512         # Output V dimension
num_blocks = 256
page_block_size = 16
topk = 8               # Number of KV tokens selected per query


def get_inputs():
    q = torch.randn(batch_size, seqlen_q, nheads, headdim_qk, dtype=torch.bfloat16)
    kv_cache = torch.randn(num_blocks, page_block_size, 1, headdim_qk,
                            dtype=torch.bfloat16)
    indices = torch.randint(0, num_blocks * page_block_size,
                            (batch_size, seqlen_q, topk), dtype=torch.int32)
    return [q, kv_cache, indices]


def get_init_inputs():
    return [nheads, headdim_qk, headdim_v, page_block_size, topk]
