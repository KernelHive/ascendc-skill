import torch
import torch.nn as nn

# Source:
# - Repo: mmprotest/mHC | Path: src/mhc/mhc.py | URL: https://github.com/mmprotest/mHC/blob/main/src/mhc/mhc.py
# - Repo: mmprotest/mHC | Path: src/mhc/norm.py | URL: https://github.com/mmprotest/mHC/blob/main/src/mhc/norm.py
# - Repo: mmprotest/mHC | Path: src/mhc/blocks.py | URL: https://github.com/mmprotest/mHC/blob/main/src/mhc/blocks.py



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.scale


class MLPResidual(nn.Module):
    """Compact residual MLP used as an optional branch function."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    """Full mHC module: compute H_pre/H_post/H_res and apply residual mixing."""

    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        tmax: int = 20,
        rms_eps: float = 1e-5,
        alpha_init: float = 0.01,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")

        self.dim = int(dim)
        self.n_streams = int(n_streams)
        self.tmax = int(tmax)

        self.norm = RMSNorm(n_streams * dim, eps=rms_eps)
        self.residual = MLPResidual(dim) if use_mlp else nn.Identity()

        self.phi_pre = nn.Linear(n_streams * dim, n_streams, bias=False)
        self.phi_post = nn.Linear(n_streams * dim, n_streams, bias=False)
        self.phi_res = nn.Linear(n_streams * dim, n_streams * n_streams, bias=False)

        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams * n_streams))

    def _sinkhorn_knopp(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.float()
        x = x - x.amax(dim=(-2, -1), keepdim=True)
        x = torch.exp(x)

        for _ in range(self.tmax):
            x = x / (x.sum(dim=-2, keepdim=True) + 1e-8)
            x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

        return x.to(dtype=logits.dtype)

    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        if x_streams.ndim != 4:
            raise ValueError("x_streams must have shape [B, T, n, C]")
        b, t, n, c = x_streams.shape
        if n != self.n_streams or c != self.dim:
            raise ValueError("shape mismatch for x_streams")

        x_flat = x_streams.reshape(b, t, n * c)
        x_norm = self.norm(x_flat)

        h_pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        h_post = self.alpha_post * self.phi_post(x_norm) + self.b_post
        h_res = self.alpha_res * self.phi_res(x_norm) + self.b_res

        h_pre = torch.sigmoid(h_pre)
        h_post = 2.0 * torch.sigmoid(h_post)
        h_res = h_res.view(b, t, n, n)
        h_res = self._sinkhorn_knopp(h_res)

        x_in = torch.einsum("btn,btnc->btc", h_pre, x_streams)
        y = self.residual(x_in)
        resmix = torch.einsum("btij,btjc->btic", h_res, x_streams)
        addy = h_post.unsqueeze(-1) * y.unsqueeze(-2)
        return resmix + addy


# Configuration
batch_size = 8
seq_len = 128
num_streams = 4
dim = 256


def get_inputs():
    x_streams = torch.randn(batch_size, seq_len, num_streams, dim)
    return [x_streams]


def get_init_inputs():
    return [dim, num_streams, 20, 1e-5, 0.01, False]
