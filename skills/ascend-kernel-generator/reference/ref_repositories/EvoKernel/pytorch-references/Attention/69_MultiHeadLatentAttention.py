import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Multi-head Latent Attention (MLA) from the DeepSeek V3 architecture.

    Compresses KV into a low-rank latent space to reduce memory access.
    The KV cache stores a single-head compressed representation of dimension
    headdim_qk (576). The query has shape (batch, seqlen_q, nheads_q, headdim_qk)
    and the output has a different head dimension headdim_v (512).

    For the attention computation:
      - K scoring uses the full headdim_qk (576) dimensions of the cache
      - V output uses the first headdim_v (512) dimensions of the cache

    Input Q:          (batch, seqlen_q, nheads_q, headdim_qk), bfloat16
    Input kv_cache:   (num_blocks, page_block_size, 1, headdim_qk), bfloat16
    Input block_table:(batch, max_num_blocks_per_seq), int32
    Input cache_seqlens: (batch,), int32
    Output:           (batch, seqlen_q, nheads_q, headdim_v), bfloat16
    """

    def __init__(self, nheads_q, headdim_qk, headdim_v, page_block_size, causal=True):
        super().__init__()
        self.nheads_q = nheads_q
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.page_block_size = page_block_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(headdim_qk)

    def _reconstruct_from_cache(self, cache, cache_seqlen, block_table_row):
        """Reconstruct contiguous KV from paged cache for one batch element."""
        seq_len = cache_seqlen.item()
        num_blocks_needed = (seq_len + self.page_block_size - 1) // self.page_block_size
        parts = []
        for block_idx in range(num_blocks_needed):
            physical_block = block_table_row[block_idx].item()
            if block_idx == num_blocks_needed - 1:
                remaining = seq_len - block_idx * self.page_block_size
                parts.append(cache[physical_block, :remaining])
            else:
                parts.append(cache[physical_block])
        return torch.cat(parts, dim=0)  # (seq_len, 1, headdim_qk)

    def forward(self, q, kv_cache, block_table, cache_seqlens):
        batch, seqlen_q, nheads_q, headdim_qk = q.shape

        outputs = []
        for b in range(batch):
            seq_len = cache_seqlens[b].item()

            # Reconstruct compressed KV: (seq_len, 1, headdim_qk)
            kv = self._reconstruct_from_cache(
                kv_cache, cache_seqlens[b], block_table[b])

            # K uses full headdim_qk dims for scoring
            # Shape: (1, 1, seq_len, headdim_qk)
            k = kv[:, :, :self.headdim_qk].unsqueeze(0).transpose(1, 2).float()
            # V uses first headdim_v dims for output
            # Shape: (1, 1, seq_len, headdim_v)
            v = kv[:, :, :self.headdim_v].unsqueeze(0).transpose(1, 2).float()

            # Broadcast single KV head to all query heads
            # k: (1, 1, seq_len, headdim_qk) -> broadcasts with (1, nheads_q, seqlen_q, headdim_qk)
            q_b = q[b:b+1].transpose(1, 2).float()  # (1, nheads_q, seqlen_q, headdim_qk)

            # Attention scores: (1, nheads_q, seqlen_q, seq_len)
            scores = torch.matmul(q_b, k.transpose(-2, -1)) * self.scale

            # Causal mask (for dense decoding mode)
            if self.causal:
                row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=q.device).unsqueeze(0)
                causal_mask = col_idx > (row_idx + seq_len - seqlen_q)
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)

            # Output: (1, nheads_q, seqlen_q, headdim_v)
            out_b = torch.matmul(attn_weights, v)
            out_b = out_b.transpose(1, 2).to(q.dtype)  # (1, seqlen_q, nheads_q, headdim_v)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)


# Configuration — DeepSeek V3 dimensions
batch_size = 4
seqlen_q = 1          # Decoding mode
nheads_q = 16
headdim_qk = 576      # DeepSeek V3 compressed QK dimension
headdim_v = 512        # DeepSeek V3 output V dimension
num_blocks = 256
page_block_size = 16
max_blocks_per_seq = 64
cache_seqlen = 512
causal = True


def get_inputs():
    q = torch.randn(batch_size, seqlen_q, nheads_q, headdim_qk, dtype=torch.bfloat16)
    # Compressed KV cache: single head, headdim_qk dims
    kv_cache = torch.randn(num_blocks, page_block_size, 1, headdim_qk,
                            dtype=torch.bfloat16)
    block_table = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq),
                                dtype=torch.int32)
    cache_seqlens = torch.full((batch_size,), cache_seqlen, dtype=torch.int32)
    return [q, kv_cache, block_table, cache_seqlens]


def get_init_inputs():
    return [nheads_q, headdim_qk, headdim_v, page_block_size, causal]
