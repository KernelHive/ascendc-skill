import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Dynamic Quantization (DyQuantize) from bfloat16 to fp8_e4m3fn.

    Supports multiple quantization granularities:
      - per-tensor: single scale for the entire tensor
      - per-token (per-channel): one scale per row
      - per-token-group: one scale per 128-element group in each row
      - per-block: one scale per 128x128 block

    Input:  (num_tokens, hidden_size), bfloat16
    Output: quantized tensor, same shape, float8_e4m3fn
            scale tensor, shape depends on granularity
    """

    FP8_MAX = 448.0  # max representable value in fp8_e4m3fn

    def __init__(self, mode='per_token_group', group_size=128):
        super().__init__()
        self.mode = mode
        self.group_size = group_size

    def _per_tensor(self, x):
        x_f = x.float()
        x_amax = x_f.abs().amax().clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = x_f / scale
        return x_scaled.clamp(-self.FP8_MAX, self.FP8_MAX), scale.view(1)

    def _per_token(self, x):
        m, n = x.shape
        x_f = x.float()
        x_amax = x_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = x_f / scale
        return x_scaled.clamp(-self.FP8_MAX, self.FP8_MAX), scale.view(m)

    def _per_token_group(self, x):
        m, n = x.shape
        assert n % self.group_size == 0
        x_f = x.float()
        x_view = x_f.view(m, -1, self.group_size)
        x_amax = x_view.abs().amax(dim=2, keepdim=True).clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = (x_view / scale).clamp(-self.FP8_MAX, self.FP8_MAX)
        return x_scaled.view(m, n), scale.view(m, -1)

    def _per_block(self, x):
        m, n = x.shape
        block = self.group_size  # 128
        # Pad to multiples of block
        m_pad = math.ceil(m / block) * block
        n_pad = math.ceil(n / block) * block
        x_padded = torch.zeros(m_pad, n_pad, dtype=torch.float32, device=x.device)
        x_padded[:m, :n] = x.float()

        x_view = x_padded.view(m_pad // block, block, n_pad // block, block)
        x_amax = x_view.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = (x_view / scale).clamp(-self.FP8_MAX, self.FP8_MAX)

        result = x_scaled.view(m_pad, n_pad)[:m, :n].contiguous()
        scale_out = scale.view(m_pad // block, n_pad // block)
        return result, scale_out

    def forward(self, x):
        if self.mode == 'per_tensor':
            return self._per_tensor(x)
        elif self.mode == 'per_token':
            return self._per_token(x)
        elif self.mode == 'per_token_group':
            return self._per_token_group(x)
        elif self.mode == 'per_block':
            return self._per_block(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# Configuration — per-token-group quantization (most common in LLM inference)
num_tokens = 2048
hidden_size = 4096
mode = 'per_token_group'
group_size = 128


def get_inputs():
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [mode, group_size]
