import torch
import torch.nn as nn

# Source:
# - Repo: Aaryyan777/mHC-Implementation | Path: mhc_kernel_fusion.py | URL: https://github.com/Aaryyan777/mHC-Implementation/blob/main/mhc_kernel_fusion.py


class FusedSinkhornKnopp(nn.Module):
    """Fused Sinkhorn-Knopp projection used in kernel fusion path."""

    def __init__(self, iterations: int = 20):
        super().__init__()
        self.iterations = int(iterations)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = torch.exp(matrix)
        for _ in range(self.iterations):
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-12)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-12)
        return matrix


class RMSNorm(nn.Module):
    """RMSNorm used in fused mapping generation."""

    def __init__(self, dim: int, eps: float = 1e-20, dtype=torch.float32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class Model(nn.Module):
    """Fused mHC kernel to compute H_pre, H_post, H_res in a single projection."""

    def __init__(self, input_dim: int, expansion_rate: int = 4, dtype=torch.float32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.expansion_rate = int(expansion_rate)
        self.n = self.expansion_rate
        self.stream_dim = self.n * self.input_dim
        self.dtype = dtype

        self.phi_params = nn.Parameter(
            torch.randn(
                self.stream_dim,
                self.n * self.n + 2 * self.n,
                dtype=dtype,
            )
            * 0.02
        )
        self.bias_params = nn.Parameter(
            torch.zeros(1, self.n * self.n + 2 * self.n, dtype=dtype)
        )

        self.alpha_pre = nn.Parameter(torch.tensor(0.01, dtype=dtype))
        self.alpha_post = nn.Parameter(torch.tensor(0.01, dtype=dtype))
        self.alpha_res = nn.Parameter(torch.tensor(0.01, dtype=dtype))

        self.rms_norm = RMSNorm(self.stream_dim, dtype=dtype)
        self.sinkhorn_knopp = FusedSinkhornKnopp(iterations=20)

    def forward(self, x_flat: torch.Tensor):
        if x_flat.ndim != 3:
            raise ValueError("x_flat must be (B, L, n*C)")
        batch_size, seq_len, _ = x_flat.shape

        x_flat = x_flat.to(self.dtype)
        x_norm = self.rms_norm(x_flat)

        all_mappings = torch.matmul(x_norm, self.phi_params)
        all_mappings = all_mappings + self.bias_params

        h_pre_tilde = all_mappings[:, :, : self.n]
        h_post_tilde = all_mappings[:, :, self.n : 2 * self.n]
        h_res_tilde = all_mappings[:, :, 2 * self.n :]
        h_res_tilde = h_res_tilde.view(batch_size, seq_len, self.n, self.n)

        h_pre_tilde = self.alpha_pre * h_pre_tilde
        h_post_tilde = self.alpha_post * h_post_tilde
        h_res_tilde = self.alpha_res * h_res_tilde

        h_pre = torch.sigmoid(h_pre_tilde)
        h_post = 2.0 * torch.sigmoid(h_post_tilde)
        h_res = self.sinkhorn_knopp(h_res_tilde)

        return h_pre, h_post, h_res


# Configuration
batch_size = 8
seq_len = 128
input_dim = 256
expansion_rate = 4


def get_inputs():
    x_flat = torch.randn(batch_size, seq_len, input_dim * expansion_rate)
    return [x_flat]


def get_init_inputs():
    return [input_dim, expansion_rate, torch.float32]
