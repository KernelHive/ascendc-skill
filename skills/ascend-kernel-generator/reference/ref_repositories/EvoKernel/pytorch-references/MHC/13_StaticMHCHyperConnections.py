import math
import torch
import torch.nn as nn

# Source:
# - Repo: tokenbender/mHC-manifold-constrained-hyper-connections | Path: hyper_connections/hyper_connections.py | URL: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections/blob/main/hyper_connections/hyper_connections.py


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 10, tau: float = 0.05) -> torch.Tensor:
    n = logits.shape[-1]
    z = logits / tau
    log_marginal = torch.full((n,), -math.log(n), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(n, device=z.device, dtype=z.dtype)
    v = torch.zeros(n, device=z.device, dtype=z.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(z + v.unsqueeze(0), dim=1)
        v = log_marginal - torch.logsumexp(z + u.unsqueeze(1), dim=0)

    return torch.exp(z + u.unsqueeze(1) + v.unsqueeze(0)) * n


def zeropower_via_newtonschulz(
    x: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs=(3.0, -3.2, 1.2),
) -> torch.Tensor:
    a, b, c = coeffs
    x = x / (x.norm() + eps)

    transpose = False
    if x.shape[0] > x.shape[1]:
        x = x.t()
        transpose = True

    for _ in range(steps):
        a_mat = x @ x.t()
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transpose:
        x = x.t()

    return x


def orthostochastic_project(
    logits: torch.Tensor,
    ns_steps: int = 5,
    ns_eps: float = 1e-7,
    ns_coeffs=(3.0, -3.2, 1.2),
) -> torch.Tensor:
    o = zeropower_via_newtonschulz(logits, steps=ns_steps, eps=ns_eps, coeffs=ns_coeffs)
    return o.square()


class Model(nn.Module):
    """Static-mHC HyperConnections with fixed H_pre/H_post/H_res logits."""

    def __init__(
        self,
        num_streams: int,
        dim: int,
        mhc_h_res_proj: str = "sinkhorn",
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coeffs=(3.0, -3.2, 1.2),
        add_branch_out_to_residual: bool = True,
        init_residual_index: int = 0,
    ):
        super().__init__()
        self.num_streams = int(num_streams)
        self.dim = int(dim)
        self.mhc_h_res_proj = mhc_h_res_proj
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tau = float(sinkhorn_tau)
        self.ns_steps = int(ns_steps)
        self.ns_eps = float(ns_eps)
        self.ns_coeffs = ns_coeffs
        self.add_branch_out_to_residual = bool(add_branch_out_to_residual)

        h_res_init = torch.full((self.num_streams, self.num_streams), -8.0)
        h_res_init.fill_diagonal_(0.0)
        self.h_res_logits = nn.Parameter(h_res_init)

        h_pre_init = torch.full((self.num_streams,), -8.0)
        h_pre_init[init_residual_index % self.num_streams] = 0.0
        self.h_pre_logits = nn.Parameter(h_pre_init)

        if self.add_branch_out_to_residual:
            self.h_post_logits = nn.Parameter(torch.zeros(self.num_streams))

        self.branch = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        if residuals.ndim != 3:
            raise ValueError("residuals must be (B, S, D)")
        bsz, streams, dim = residuals.shape
        if streams != self.num_streams or dim != self.dim:
            raise ValueError("shape mismatch for residuals")

        if self.mhc_h_res_proj == "orthostochastic":
            h_res = orthostochastic_project(
                self.h_res_logits, self.ns_steps, self.ns_eps, self.ns_coeffs
            )
        else:
            h_res = sinkhorn_log(
                self.h_res_logits, self.sinkhorn_iters, self.sinkhorn_tau
            )

        h_pre = torch.softmax(self.h_pre_logits, dim=-1)
        h_post = None
        if self.add_branch_out_to_residual:
            h_post = torch.softmax(self.h_post_logits, dim=-1)

        residuals_mixed = torch.einsum("st,bsd->btd", h_res, residuals)
        branch_input = torch.einsum("s,bsd->bd", h_pre, residuals)
        branch_output = self.branch(branch_input)

        if h_post is None:
            return branch_output

        depth_out = torch.einsum("bd,s->bsd", branch_output, h_post)
        return residuals_mixed + depth_out


# Configuration
batch_size = 8
num_streams = 4
dim = 256


def get_inputs():
    residuals = torch.randn(batch_size, num_streams, dim)
    return [residuals]


def get_init_inputs():
    return [num_streams, dim, "sinkhorn", 10, 0.05, 5, 1e-7, (3.0, -3.2, 1.2), True, 0]
