import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE) for LLM attention.

    Injects relative position information into Query and Key vectors using
    rotation matrices derived from sinusoidal position encodings.
    Supports NeoX-style rotation (split-half).

    Input query:     (num_tokens, num_heads, head_dim), bfloat16
    Input key:       (num_tokens, num_heads, head_dim), bfloat16
    Input positions: (num_tokens,), int32
    Output:          rotated query and key with same shapes, bfloat16
    """

    def __init__(self, head_dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Precompute sincos cache: (max_seq_len, head_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        # Cache stores [cos, sin] concatenated: (max_seq_len, head_dim)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    @staticmethod
    def _rotate_neox(x):
        """NeoX-style rotation: split in half and rotate."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, query, key, positions):
        # Gather cos/sin for the given positions: (num_tokens, head_dim//2)
        cos = self.cos_cache[positions.long()]  # (num_tokens, head_dim//2)
        sin = self.sin_cache[positions.long()]  # (num_tokens, head_dim//2)

        # Expand to full head_dim by repeating: (num_tokens, head_dim)
        cos = cos.repeat(1, 2).unsqueeze(1)  # (num_tokens, 1, head_dim)
        sin = sin.repeat(1, 2).unsqueeze(1)  # (num_tokens, 1, head_dim)

        # Apply rotation
        query_rot = query.float() * cos + self._rotate_neox(query.float()) * sin
        key_rot = key.float() * cos + self._rotate_neox(key.float()) * sin

        return query_rot.to(query.dtype), key_rot.to(key.dtype)


# Configuration — realistic LLM dimensions
num_tokens = 2048
num_heads = 32
head_dim = 128
max_seq_len = 8192


def get_inputs():
    query = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    positions = torch.randint(0, max_seq_len, (num_tokens,), dtype=torch.int32)
    return [query, key, positions]


def get_init_inputs():
    return [head_dim, max_seq_len]
