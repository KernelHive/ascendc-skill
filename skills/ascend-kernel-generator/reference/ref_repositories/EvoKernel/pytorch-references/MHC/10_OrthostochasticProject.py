import torch
import torch.nn as nn

# Source:
# - Repo: tokenbender/mHC-manifold-constrained-hyper-connections | Path: hyper_connections/hyper_connections.py | URL: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections/blob/main/hyper_connections/hyper_connections.py


class Model(nn.Module):
    """Orthostochastic projection via Newton–Schulz zero-power iteration."""

    def __init__(self, steps: int = 5, eps: float = 1e-7, coeffs=(3.0, -3.2, 1.2)):
        super().__init__()
        self.steps = int(steps)
        self.eps = float(eps)
        self.coeffs = coeffs

    def _zeropower_via_newtonschulz(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("logits must be 2D (n, n)")

        a, b, c = self.coeffs
        x = x / (x.norm() + self.eps)

        transpose = False
        if x.shape[0] > x.shape[1]:
            x = x.t()
            transpose = True

        for _ in range(self.steps):
            a_mat = x @ x.t()
            b_mat = b * a_mat + c * a_mat @ a_mat
            x = a * x + b_mat @ x

        if transpose:
            x = x.t()

        return x

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        o = self._zeropower_via_newtonschulz(logits)
        return o.square()


# Configuration
num_streams = 8


def get_inputs():
    logits = torch.randn(num_streams, num_streams)
    return [logits]


def get_init_inputs():
    return [5, 1e-7, (3.0, -3.2, 1.2)]
