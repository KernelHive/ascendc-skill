import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


# -----------------------------------------------------------------------------
# CUDA/C++ extension
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -----------------------------------------------------------------------------
// Fused causal attention (same as reference example): q,k,v -> y
// One warp computes one (b,h,t) row; lanes own output dims d=lane,lane+32,...
// -----------------------------------------------------------------------------
__global__ void causal_attn_fwd_warp_kernel(
    const float* __restrict__ q,  // [B, H, T, HS]
    const float* __restrict__ k,  // [B, H, T, HS]
    const float* __restrict__ v,  // [B, H, T, HS]
    float* __restrict__ y,        // [B, H, T, HS]
    int B, int H, int T, int HS,
    float scale,
    bool causal
) {
    int row = (int)blockIdx.x; // one row per block (one warp)
    int lane = threadIdx.x;    // 0..31
    if (lane >= 32) return;

    int bh = row / T;
    int t  = row - bh * T;
    int b  = bh / H;
    int h  = bh - b * H;

    if (b >= B || h >= H || t >= T) return;

    const int64_t base_q = (((int64_t)b * H + h) * T + t) * (int64_t)HS;
    const float* qptr = q + base_q;

    float row_max = -INFINITY;
    int j_max = causal ? (t + 1) : T;
    for (int j = lane; j < j_max; j += 32) {
        const int64_t base_k = (((int64_t)b * H + h) * T + j) * (int64_t)HS;
        const float* kptr = k + base_k;

        float dot = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < HS; d++) dot += qptr[d] * kptr[d];
        float logit = dot * scale;
        row_max = fmaxf(row_max, logit);
    }
    row_max = warp_reduce_max(row_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float denom = 0.0f;
    for (int j = lane; j < j_max; j += 32) {
        const int64_t base_k = (((int64_t)b * H + h) * T + j) * (int64_t)HS;
        const float* kptr = k + base_k;

        float dot = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < HS; d++) dot += qptr[d] * kptr[d];
        float logit = dot * scale;
        float w = __expf(logit - row_max);
        denom += w;
    }
    denom = warp_reduce_sum(denom);
    denom = __shfl_sync(0xffffffff, denom, 0);

    for (int d = lane; d < HS; d += 32) {
        float num = 0.0f;
        for (int j = 0; j < j_max; j++) {
            const int64_t base_k = (((int64_t)b * H + h) * T + j) * (int64_t)HS;
            const float* kptr = k + base_k;
            const float* vptr = v + base_k;

            float dot = 0.0f;
            #pragma unroll 1
            for (int dd = 0; dd < HS; dd++) dot += qptr[dd] * kptr[dd];
            float logit = dot * scale;
            float w = __expf(logit - row_max);
            num += w * vptr[d];
        }
        y[base_q + d] = num / denom;
    }
}

torch::Tensor min_gpt_causal_attention_cuda(torch::Tensor q,
                                           torch::Tensor k,
                                           torch::Tensor v,
                                           bool causal) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be [B,H,T,HS]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q,k,v must have same shape");

    int B  = (int)q.size(0);
    int H  = (int)q.size(1);
    int T  = (int)q.size(2);
    int HS = (int)q.size(3);
    TORCH_CHECK(HS > 0 && HS <= 256, "HS must be in (0,256]");

    auto y = torch::empty_like(q);
    float scale = 1.0f / sqrtf((float)HS);

    int rows = B * H * T;
    dim3 grid(rows);
    dim3 block(32);

    causal_attn_fwd_warp_kernel<<<grid, block>>>(
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        B, H, T, HS, scale, causal
    );
    return y;
}

// -----------------------------------------------------------------------------
// LayerNorm forward (x [N,C], gamma [C], beta [C]) -> y [N,C]
// One block per row; power-of-two threads; shared-memory reductions.
// -----------------------------------------------------------------------------
__global__ void layernorm_fwd_kernel(
    const float* __restrict__ x,     // [N, C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ y,           // [N, C]
    int N, int C,
    float eps
) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (row >= N) return;

    extern __shared__ float smem[];
    float* ssum = smem;
    float* ssq  = smem + blockDim.x;

    const int64_t base = (int64_t)row * (int64_t)C;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float v = x[base + c];
        sum += v;
        sumsq += v * v;
    }

    ssum[tid] = sum;
    ssq[tid] = sumsq;
    __syncthreads();

    // Reduce within block (assumes power-of-two threads)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
            ssq[tid]  += ssq[tid + stride];
        }
        __syncthreads();
    }

    float mean = ssum[0] / (float)C;
    float var  = ssq[0] / (float)C - mean * mean;
    var = fmaxf(var, 0.0f);
    float invstd = rsqrtf(var + eps);

    for (int c = tid; c < C; c += blockDim.x) {
        float v = x[base + c];
        float n = (v - mean) * invstd;
        y[base + c] = n * gamma[c] + beta[c];
    }
}

torch::Tensor layernorm_fwd_cuda(torch::Tensor x2d,
                                torch::Tensor gamma,
                                torch::Tensor beta,
                                double eps) {
    CHECK_INPUT(x2d);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(x2d.dim() == 2, "x must be 2D [N,C]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D [C]");
    TORCH_CHECK(x2d.size(1) == gamma.size(0) && gamma.size(0) == beta.size(0), "C mismatch");

    int N = (int)x2d.size(0);
    int C = (int)x2d.size(1);

    auto y = torch::empty_like(x2d);

    // Power-of-two threads for reduction
    int threads = 256;
    if (C <= 128) threads = 128;
    if (C <= 64)  threads = 64;
    if (C <= 32)  threads = 32;
    // ensure power-of-two
    TORCH_CHECK((threads & (threads - 1)) == 0, "threads must be power-of-two");

    dim3 grid(N);
    dim3 block(threads);
    size_t shmem = 2 * (size_t)threads * sizeof(float);

    layernorm_fwd_kernel<<<grid, block, shmem>>>(
        (const float*)x2d.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, C,
        (float)eps
    );
    return y;
}

// -----------------------------------------------------------------------------
// Residual add in-place: x += y
// -----------------------------------------------------------------------------
__global__ void residual_add_inplace_kernel(float* __restrict__ x,
                                           const float* __restrict__ y,
                                           int64_t n) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i < n) x[i] += y[i];
}

void residual_add_inplace_cuda(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have same shape");

    int64_t n = x.numel();
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    residual_add_inplace_kernel<<<(unsigned int)blocks, threads>>>(
        (float*)x.data_ptr<float>(),
        (const float*)y.data_ptr<float>(),
        n
    );
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor min_gpt_causal_attention_cuda(torch::Tensor q,
                                           torch::Tensor k,
                                           torch::Tensor v,
                                           bool causal);

torch::Tensor layernorm_fwd_cuda(torch::Tensor x2d,
                                torch::Tensor gamma,
                                torch::Tensor beta,
                                double eps);

void residual_add_inplace_cuda(torch::Tensor x, torch::Tensor y);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mini_gpt_block",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "min_gpt_causal_attention_cuda",
        "layernorm_fwd_cuda",
        "residual_add_inplace_cuda",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd


# -----------------------------------------------------------------------------
# New model using custom CUDA ops for the block
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """Transformer block with CUDA fast-path for LN + attention core + residual adds (inference, no dropout)."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
        self.custom_ops = custom_ops_lib
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

    def _fast_contract(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew fast path expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._fast_contract(x)

        # Only enable custom kernels on inference/no-dropout for semantics match
        fast_ok = (not self.training) and (self.attn_pdrop == 0.0) and (self.resid_pdrop == 0.0)

        B, T, C = x.shape
        nh = self.n_head
        hs = C // nh

        if not fast_ok:
            # Reference path (PyTorch)
            x = x + self._attn_ref(self.ln_1(x))
            x = x + self.mlpf(self.ln_2(x))
            return x

        # --- LN1 (custom) ---
        x2d = x.view(B * T, C).contiguous()
        ln1_2d = self.custom_ops.layernorm_fwd_cuda(
            x2d, self.ln_1.weight.contiguous(), self.ln_1.bias.contiguous(), float(self.ln_1.eps)
        )
        ln1 = ln1_2d.view(B, T, C)

        # --- Attention projections (PyTorch) ---
        qkv = self.attn.c_attn(ln1)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, nh, hs).transpose(1, 2).contiguous()  # [B, nh, T, hs]
        q = q.view(B, T, nh, hs).transpose(1, 2).contiguous()
        v = v.view(B, T, nh, hs).transpose(1, 2).contiguous()

        # --- Attention core (custom) ---
        y = self.custom_ops.min_gpt_causal_attention_cuda(q, k, v, True)  # [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --- Output projection (PyTorch) ---
        y = self.attn.c_proj(y).contiguous()

        # --- Residual add (custom in-place) ---
        x = x.contiguous()
        self.custom_ops.residual_add_inplace_cuda(x, y)

        # --- LN2 (custom) ---
        x2d = x.view(B * T, C).contiguous()
        ln2_2d = self.custom_ops.layernorm_fwd_cuda(
            x2d, self.ln_2.weight.contiguous(), self.ln_2.bias.contiguous(), float(self.ln_2.eps)
        )
        ln2 = ln2_2d.view(B, T, C)

        # --- MLP (PyTorch) ---
        m = self.mlp
        mlp_out = m.c_proj(m.act(m.c_fc(ln2))).contiguous()

        # --- Residual add (custom in-place) ---
        self.custom_ops.residual_add_inplace_cuda(x, mlp_out)
        return x

    def _attn_ref(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.attn.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.attn.resid_dropout(self.attn.c_proj(y))
        return y


# -----------------------------------------------------------------------------
# Keep same helper signatures
# -----------------------------------------------------------------------------
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]