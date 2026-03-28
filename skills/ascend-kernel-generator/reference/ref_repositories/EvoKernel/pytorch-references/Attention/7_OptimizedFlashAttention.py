import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OptimizedFlashAttention(nn.Module):
    """
    Optimized Flash Attention with multiple memory and compute optimizations,
    including optional RoPE and ALiBi positional encodings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_rope: bool = True,
        use_alibi: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.use_rope = use_rope
        self.use_alibi = use_alibi

        if use_rope:
            self._init_rope()
        if use_alibi:
            self._init_alibi()

    def _init_rope(self):
        """Initialize RoPE inverse frequency buffer."""
        dim = self.d_k
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('rope_inv_freq', inv_freq)

    def _init_alibi(self):
        """Initialize ALiBi slope buffer."""
        slopes = torch.tensor([
            2 ** (-8 * i / self.n_heads) for i in range(self.n_heads)
        ])
        self.register_buffer('alibi_slopes', slopes)

    def apply_rope(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Rotary Position Embedding (RoPE).

        Args:
            x: [batch_size, n_heads, seq_len, d_k]
            position_ids: [batch_size, seq_len]

        Returns:
            rotated x with same shape
        """
        sincos = torch.einsum('bi,j->bij', position_ids.float(), self.rope_inv_freq)
        sin = sincos.sin().repeat_interleave(2, dim=-1)
        cos = sincos.cos().repeat_interleave(2, dim=-1)

        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        x = x * cos.unsqueeze(1) + x_rot * sin.unsqueeze(1)

        return x

    def _compute_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute ALiBi positional bias.

        Returns:
            bias: [n_heads, seq_len_q, seq_len_k]
        """
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        relative_pos = -(q_pos - k_pos).abs()

        alibi_bias = self.alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos.unsqueeze(0)
        return alibi_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE positional encoding
        if self.use_rope:
            Q = self.apply_rope(Q, position_ids)
            K = self.apply_rope(K, position_ids)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply ALiBi bias
        if self.use_alibi:
            position_bias = self._compute_alibi_bias(seq_len, seq_len, x.device)
            scores = scores + position_bias.unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class Model(nn.Module):
    """
    Benchmark wrapper for OptimizedFlashAttention.
    Implements optimized Flash Attention with RoPE and optional ALiBi encodings.
    """

    def __init__(self, d_model, n_heads, use_rope, use_alibi):
        super().__init__()
        self.attn = OptimizedFlashAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_rope=use_rope,
            use_alibi=use_alibi,
        )

    def forward(self, x):
        return self.attn(x)


# Configuration
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
use_rope = True
use_alibi = False


def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]


def get_init_inputs():
    return [d_model, n_heads, use_rope, use_alibi]
