import torch
import torch.nn as nn

# Source:
# - Repo: tile-ai/tilelang | Path: examples/deepseek_mhc/example_mhc_pre.py | URL: https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc/example_mhc_pre.py


def sinkhorn_normalize(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """
    Sinkhorn-Knopp normalization: alternating row/column normalization
    to produce a doubly stochastic matrix.

    Args:
        x: Input tensor of shape (..., hc_mult, hc_mult)
        repeat: Number of sinkhorn iterations
        eps: Small epsilon for numerical stability

    Returns:
        Doubly stochastic matrix of same shape
    """
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


class Model(nn.Module):
    """
    DeepSeek mHC (multi-head chunked) pre-block operator from tilelang examples.

    Implements the pre-processing stage of multi-head chunked computation:
    1. Fused GEMM + square-sum: computes mixes = RMSNorm(residual_flat) @ fn^T
    2. Split mixes into pre_mix, post_mix, and comb_mix (res_mix)
    3. Apply sigmoid activations and sinkhorn normalization on comb_mix
    4. Compute layer_input as weighted sum of residual streams

    Args:
        hidden_size: Hidden dimension per stream
        hc_mult: Number of streams (heads), typically 4
        rms_eps: Epsilon for RMS normalization
        hc_pre_eps: Epsilon added to pre_mix after sigmoid
        hc_sinkhorn_eps: Epsilon for sinkhorn normalization
        hc_post_mult_value: Multiplier for post_mix after sigmoid
        sinkhorn_repeat: Number of sinkhorn normalization iterations
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.rms_eps = rms_eps
        self.hc_pre_eps = hc_pre_eps
        self.hc_sinkhorn_eps = hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat

        self.hc_mult2 = hc_mult * hc_mult
        self.hc_mult3 = hc_mult * 2 + self.hc_mult2

    def forward(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple:
        """
        Forward pass for mHC pre block.

        Args:
            residual: shape (num_tokens, hc_mult, hidden_size), dtype bfloat16
            fn: shape (hc_mult3, hc_mult * hidden_size), dtype float32
            hc_scale: shape (3,), dtype float32
            hc_base: shape (hc_mult3,), dtype float32

        Returns:
            post_mix: shape (num_tokens, hc_mult, 1), dtype float32
            comb_mix: shape (num_tokens, hc_mult, hc_mult), dtype float32
            layer_input: shape (num_tokens, hidden_size), dtype bfloat16
        """
        hc_mult = self.hc_mult

        # Flatten residual across streams: (N, hc_mult, H) -> (N, hc_mult * H)
        residual_flat = residual.flatten(-2, -1).float()

        # Compute squared sum for RMS normalization
        sqrsum = residual_flat.square().sum(-1)

        # Fused RMSNorm + linear projection: mixes = RMSNorm(residual_flat) @ fn^T
        rms_inv = (sqrsum.unsqueeze(-1) / fn.shape[-1] + self.rms_eps).rsqrt()
        mixes = residual_flat @ fn.T * rms_inv

        # Expand hc_scale to match mixes dimensions and apply affine transform
        hc_scale_expanded = torch.cat([
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ])
        mixes = mixes * hc_scale_expanded + hc_base

        # Split mixes into three components
        pre_mix = mixes[..., :hc_mult].sigmoid().unsqueeze(-1) + self.hc_pre_eps
        post_mix = (
            mixes[..., hc_mult : 2 * hc_mult].sigmoid() * self.hc_post_mult_value
        ).unsqueeze(-1)
        res_mix = mixes[..., 2 * hc_mult :].view(-1, hc_mult, hc_mult)

        # Sinkhorn normalization on combination mix
        comb_mix = sinkhorn_normalize(
            res_mix, repeat=self.sinkhorn_repeat, eps=self.hc_sinkhorn_eps
        )

        # Weighted sum of residual streams to produce layer input
        layer_input = (residual * pre_mix).sum(-2).bfloat16()

        return post_mix, comb_mix, layer_input


# Configuration - matches DeepSeek mHC typical dimensions
num_tokens = 2048
hidden_size = 2560
hc_mult = 4
rms_eps = 1e-6
hc_pre_eps = 1e-6
hc_sinkhorn_eps = 1e-6
hc_post_mult_value = 1.0
sinkhorn_repeat = 10

hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # = 24 for hc_mult=4


def get_inputs():
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16
    )
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32) * 1e-4
    hc_scale = torch.randn(3, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32) * 0.1
    return [residual, fn, hc_scale, hc_base]


def get_init_inputs():
    return [
        hidden_size,
        hc_mult,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    ]
