import torch
import torch.nn as nn

# Source:
# - Repo: MarcoDotIO/mhc-deepseek-implementation | Path: src/mhc/mhc.py | URL: https://github.com/MarcoDotIO/mhc-deepseek-implementation/blob/main/src/mhc/mhc.py



class Model(nn.Module):
    """Compute H_pre, H_post, and H_res mappings for mHC streams."""

    def __init__(
        self,
        n_streams: int,
        hidden_dim: int,
        tmax: int = 20,
        alpha_init: float = 0.01,
        rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.n = int(n_streams)
        self.c = int(hidden_dim)
        self.tmax = int(tmax)
        self.eps = 1e-8

        flat_dim = self.n * self.c
        self.rmsnorm_eps = float(rmsnorm_eps)

        self.phi_pre = nn.Parameter(torch.empty(flat_dim, self.n))
        self.phi_post = nn.Parameter(torch.empty(flat_dim, self.n))
        self.phi_res = nn.Parameter(torch.empty(flat_dim, self.n * self.n))

        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.02
        nn.init.normal_(self.phi_pre, mean=0.0, std=std)
        nn.init.normal_(self.phi_post, mean=0.0, std=std)
        nn.init.normal_(self.phi_res, mean=0.0, std=std)
        nn.init.zeros_(self.b_pre)
        nn.init.zeros_(self.b_post)
        nn.init.zeros_(self.b_res)

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        return x_float * torch.rsqrt(rms + self.rmsnorm_eps)

    def _sinkhorn_knopp(self, logits: torch.Tensor) -> torch.Tensor:
        log_m = logits.float()
        log_m = log_m - log_m.amax(dim=(-2, -1), keepdim=True)
        for _ in range(self.tmax):
            log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
            log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
        return torch.exp(log_m)

    def forward(self, x_stream: torch.Tensor):
        if x_stream.ndim != 4:
            raise ValueError("x_stream must be (B,T,n,C)")
        b, t, n, c = x_stream.shape
        if n != self.n or c != self.c:
            raise ValueError("shape mismatch for streams")

        x_flat = x_stream.reshape(b * t, n * c)
        x_norm = self._rmsnorm(x_flat)

        h_pre_tilde = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre
        h_post_tilde = self.alpha_post * (x_norm @ self.phi_post) + self.b_post

        h_res_dyn = x_norm @ self.phi_res
        h_res_tilde = self.alpha_res * h_res_dyn.reshape(b * t, n, n) + self.b_res

        h_pre = torch.sigmoid(h_pre_tilde).reshape(b, t, n)
        h_post = (2.0 * torch.sigmoid(h_post_tilde)).reshape(b, t, n)
        h_res = self._sinkhorn_knopp(h_res_tilde.reshape(b, t, n, n))

        return h_pre, h_post, h_res


# Configuration
batch_size = 8
seq_len = 128
num_streams = 4
dim = 256


def get_inputs():
    x_stream = torch.randn(batch_size, seq_len, num_streams, dim)
    return [x_stream]


def get_init_inputs():
    return [num_streams, dim, 20, 0.01, 1e-6]
