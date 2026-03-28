import torch
import torch.nn as nn

# Source:
# - Repo: MarcoDotIO/mhc-deepseek-implementation | Path: src/mhc/sinkhorn.py | URL: https://github.com/MarcoDotIO/mhc-deepseek-implementation/blob/main/src/mhc/sinkhorn.py



class Model(nn.Module):
    """Log-space Sinkhorn-Knopp normalization for doubly stochastic matrices."""

    def __init__(self, tmax: int = 20, eps: float = 1e-8, clamp_min: float = 0.0):
        super().__init__()
        self.tmax = int(tmax)
        self.eps = float(eps)
        self.clamp_min = float(clamp_min)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        log_m = logits.float()
        log_m = log_m - log_m.amax(dim=(-2, -1), keepdim=True)

        for _ in range(self.tmax):
            log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
            log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)

        m = torch.exp(log_m)

        if self.clamp_min > 0.0:
            m = m.clamp_min(self.clamp_min)
            m = m / (m.sum(dim=-1, keepdim=True) + self.eps)
            m = m / (m.sum(dim=-2, keepdim=True) + self.eps)

        return m


# Configuration
batch_size = 16
num_streams = 8


def get_inputs():
    logits = torch.randn(batch_size, num_streams, num_streams)
    return [logits]


def get_init_inputs():
    return [20, 1e-8, 0.0]
