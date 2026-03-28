import torch
import torch.nn as nn

# Source:
# - Repo: tile-ai/tilelang | Path: examples/deepseek_mhc/example_mhc_post.py | URL: https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mhc/example_mhc_post.py


class Model(nn.Module):
    """
    DeepSeek mHC (multi-head chunked) post-block operator from tilelang examples.

    Implements the post-processing stage of multi-head chunked computation:
    given the layer output x, residual streams, post_layer_mix weights, and
    comb_res_mix (doubly stochastic combination matrix from sinkhorn), computes
    the updated residual streams as:

        output = x * post_layer_mix + comb_res_mix^T @ residual

    where:
    - x is broadcast across streams and scaled by post_layer_mix
    - comb_res_mix^T @ residual mixes the existing residual streams

    Args:
        hc_mult: Number of streams (heads), typically 4
        hidden_size: Hidden dimension per stream
    """

    def __init__(self, hc_mult: int, hidden_size: int):
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for mHC post block.

        Args:
            x: Layer output, shape (num_tokens, hidden_size), dtype bfloat16
            residual: Residual streams, shape (num_tokens, hc_mult, hidden_size), dtype bfloat16
            post_layer_mix: Per-stream scaling for layer output,
                            shape (num_tokens, hc_mult, 1), dtype float32
            comb_res_mix: Combination matrix (doubly stochastic),
                          shape (num_tokens, hc_mult, hc_mult), dtype float32

        Returns:
            Updated residual streams, shape (num_tokens, hc_mult, hidden_size), dtype bfloat16
        """
        # Mix existing residual streams via transposed combination matrix
        # comb_res_mix.mT: (N, hc_mult, hc_mult), residual.float(): (N, hc_mult, H)
        # term2: (N, hc_mult, H) - linear combination of residual streams
        term2 = torch.bmm(comb_res_mix.mT, residual.float())

        # Broadcast layer output across streams, scale by post_layer_mix, add mixed residuals
        # x.float().unsqueeze(-2): (N, 1, H), post_layer_mix: (N, hc_mult, 1)
        # Result: (N, hc_mult, H)
        output = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()

        return output


# Configuration - matches DeepSeek mHC typical dimensions
num_tokens = 4096
hidden_size = 2560
hc_mult = 4


def get_inputs():
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32)
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    return [hc_mult, hidden_size]
