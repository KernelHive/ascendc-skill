import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Paged KV Cache Attention for LLM inference decoding.

    Uses paged (block-based) memory management for the KV cache, supporting
    Grouped Query Attention (GQA). Each batch element has a page table mapping
    logical block indices to physical blocks in the cache.

    Input Q:          (batch, seqlen_q, nheads_q, headdim), bfloat16
    Input k_cache:    (num_blocks, page_block_size, nheads_kv, headdim), bfloat16
    Input v_cache:    (num_blocks, page_block_size, nheads_kv, headdim), bfloat16
    Input cache_seqlens: (batch,), int32
    Input page_table: (batch, max_num_blocks_per_seq), int32
    Output:           (batch, seqlen_q, nheads_q, headdim), bfloat16
    """

    def __init__(self, nheads_q, nheads_kv, headdim, page_block_size, causal=True):
        super().__init__()
        self.nheads_q = nheads_q
        self.nheads_kv = nheads_kv
        self.headdim = headdim
        self.page_block_size = page_block_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(headdim)
        assert nheads_q % nheads_kv == 0
        self.n_groups = nheads_q // nheads_kv

    def _reconstruct_kv(self, cache, cache_seqlen, page_table_row):
        """Reconstruct contiguous KV from paged cache for a single batch element."""
        seq_len = cache_seqlen.item()
        num_blocks_needed = (seq_len + self.page_block_size - 1) // self.page_block_size
        kv_parts = []
        for block_idx in range(num_blocks_needed):
            physical_block = page_table_row[block_idx].item()
            if block_idx == num_blocks_needed - 1:
                remaining = seq_len - block_idx * self.page_block_size
                kv_parts.append(cache[physical_block, :remaining])
            else:
                kv_parts.append(cache[physical_block])
        return torch.cat(kv_parts, dim=0)  # (seq_len, nheads_kv, headdim)

    def forward(self, q, k_cache, v_cache, cache_seqlens, page_table):
        batch = q.shape[0]
        seqlen_q = q.shape[1]

        outputs = []
        for b in range(batch):
            seq_len = cache_seqlens[b].item()

            # Reconstruct K, V from paged cache
            k = self._reconstruct_kv(k_cache, cache_seqlens[b], page_table[b])
            v = self._reconstruct_kv(v_cache, cache_seqlens[b], page_table[b])

            # Reshape for attention: (1, nheads_kv, seq_len, headdim)
            k = k.unsqueeze(0).transpose(1, 2).float()
            v = v.unsqueeze(0).transpose(1, 2).float()

            # GQA: expand KV heads to match Q heads
            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)

            q_b = q[b:b+1].transpose(1, 2).float()  # (1, nheads_q, seqlen_q, headdim)

            # Attention scores
            scores = torch.matmul(q_b, k.transpose(-2, -1)) * self.scale

            # Causal mask
            if self.causal:
                row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=q.device).unsqueeze(0)
                causal_mask = col_idx > (row_idx + seq_len - seqlen_q)
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)
            out_b = torch.matmul(attn_weights, v)
            out_b = out_b.transpose(1, 2).to(q.dtype)  # (1, seqlen_q, nheads_q, headdim)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)


# Configuration — inference decoding scenario
batch_size = 4
seqlen_q = 1        # Decoding: query length = 1
nheads_q = 16
nheads_kv = 8
headdim = 128
num_blocks = 256
page_block_size = 16
max_blocks_per_seq = 64
cache_seqlen = 512   # Cached context length per request
causal = True


def get_inputs():
    q = torch.randn(batch_size, seqlen_q, nheads_q, headdim, dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, page_block_size, nheads_kv, headdim,
                           dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, page_block_size, nheads_kv, headdim,
                           dtype=torch.bfloat16)
    cache_seqlens = torch.full((batch_size,), cache_seqlen, dtype=torch.int32)
    page_table = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq),
                               dtype=torch.int32)
    return [q, k_cache, v_cache, cache_seqlens, page_table]


def get_init_inputs():
    return [nheads_q, nheads_kv, headdim, page_block_size, causal]
