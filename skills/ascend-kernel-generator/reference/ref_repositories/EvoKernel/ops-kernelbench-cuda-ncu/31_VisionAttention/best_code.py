import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA: residual + LayerNorm forward for float32 contiguous [N, C]
# - Hot-path specialization for C==128:
#   * warp-per-row, no __syncthreads
#   * float4 vectorized loads/stores (coalesced)
#   * keep v=x+r in registers (reduce global traffic: no second read of x/r)
#   * correct warp broadcast using __shfl_sync(srcLane=0)
# - Fallback: baseline block-Welford for general C (correct and stable)
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float shfl_down(float v, int offset) {
    return __shfl_down_sync(0xffffffffu, v, offset);
}
__device__ __forceinline__ float shfl_broadcast(float v, int src_lane) {
    return __shfl_sync(0xffffffffu, v, src_lane);
}

__device__ __forceinline__ void welford_combine(float &mean, float &m2, float &count,
                                               float mean_b, float m2_b, float count_b) {
    if (count_b == 0.0f) return;
    if (count == 0.0f) { mean = mean_b; m2 = m2_b; count = count_b; return; }
    float delta = mean_b - mean;
    float tot = count + count_b;
    mean = mean + delta * (count_b / tot);
    m2 = m2 + m2_b + delta * delta * (count * count_b / tot);
    count = tot;
}

__device__ __forceinline__ void warp_welford_reduce(float &mean, float &m2, float &count) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mean_b  = shfl_down(mean, offset);
        float m2_b    = shfl_down(m2, offset);
        float count_b = shfl_down(count, offset);
        welford_combine(mean, m2, count, mean_b, m2_b, count_b);
    }
}

// -------------------- Fast path: C == 128 --------------------
// 1 warp computes 1 row. Keep v=x+r in registers and reuse for output.
// Layout is [N, 128] contiguous float. We use float4 (32 floats/warp iteration).
__global__ __launch_bounds__(128, 6)
void residual_layernorm_c128_warp_fwd(
    const float* __restrict__ x,        // [N, 128]
    const float* __restrict__ r,        // [N, 128]
    const float* __restrict__ gamma,    // [128]
    const float* __restrict__ beta,     // [128]
    float* __restrict__ y,              // [N, 128]
    int N,
    float eps
){
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;
    int warps_per_block = (int)blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= N) return;

    const float4* __restrict__ x4 = (const float4*)(x + ((int64_t)row << 7)); // row*128
    const float4* __restrict__ r4 = (const float4*)(r + ((int64_t)row << 7));
    const float4* __restrict__ g4 = (const float4*)gamma;
    const float4* __restrict__ b4 = (const float4*)beta;
    float4* __restrict__ y4 = (float4*)(y + ((int64_t)row << 7));

    // Each lane handles one float4 at index = lane (0..31) => 32*4=128 elements.
    // Store v in registers for reuse in output.
    float4 xv = x4[lane];
    float4 rv = r4[lane];
    float v0 = xv.x + rv.x;
    float v1 = xv.y + rv.y;
    float v2 = xv.z + rv.z;
    float v3 = xv.w + rv.w;

    // Welford over 4 values per lane
    float mean = 0.0f, m2 = 0.0f, cnt = 0.0f;

    cnt += 1.0f; { float d = v0 - mean; mean += d / cnt; float d2 = v0 - mean; m2 += d * d2; }
    cnt += 1.0f; { float d = v1 - mean; mean += d / cnt; float d2 = v1 - mean; m2 += d * d2; }
    cnt += 1.0f; { float d = v2 - mean; mean += d / cnt; float d2 = v2 - mean; m2 += d * d2; }
    cnt += 1.0f; { float d = v3 - mean; mean += d / cnt; float d2 = v3 - mean; m2 += d * d2; }

    // Reduce across warp (32 lanes => 128 samples)
    warp_welford_reduce(mean, m2, cnt);

    // Broadcast final stats from lane 0
    float final_mean = shfl_broadcast(mean, 0);
    float final_m2   = shfl_broadcast(m2, 0);
    float final_cnt  = shfl_broadcast(cnt, 0);

    float var = (final_cnt > 0.0f) ? (final_m2 / final_cnt) : 0.0f;
    float inv_std = rsqrtf(var + eps);

    // Affine using float4
    float4 gv = g4[lane];
    float4 bv = b4[lane];

    float n0 = (v0 - final_mean) * inv_std;
    float n1 = (v1 - final_mean) * inv_std;
    float n2 = (v2 - final_mean) * inv_std;
    float n3 = (v3 - final_mean) * inv_std;

    float4 out;
    out.x = fmaf(n0, gv.x, bv.x);
    out.y = fmaf(n1, gv.y, bv.y);
    out.z = fmaf(n2, gv.z, bv.z);
    out.w = fmaf(n3, gv.w, bv.w);

    y4[lane] = out;
}

// -------------------- Fallback: general C (baseline) --------------------
__global__ __launch_bounds__(256, 2)
void residual_layernorm_block_welford_fwd(
    const float* __restrict__ x,        // [N, C]
    const float* __restrict__ r,        // [N, C]
    const float* __restrict__ gamma,    // [C]
    const float* __restrict__ beta,     // [C]
    float* __restrict__ y,              // [N, C]
    int N, int C,
    float eps
){
    int row = (int)blockIdx.x;
    if (row >= N) return;

    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    const float* xrow = x + (int64_t)row * (int64_t)C;
    const float* rrow = r + (int64_t)row * (int64_t)C;
    float* yrow       = y + (int64_t)row * (int64_t)C;

    float mean = 0.0f;
    float m2   = 0.0f;
    float cnt  = 0.0f;

    for (int c = tid; c < C; c += blockDim.x) {
        float v = xrow[c] + rrow[c];
        cnt += 1.0f;
        float delta = v - mean;
        mean += delta / cnt;
        float delta2 = v - mean;
        m2 += delta * delta2;
    }

    // warp reduce
    warp_welford_reduce(mean, m2, cnt);

    __shared__ float sh_mean[8];
    __shared__ float sh_m2[8];
    __shared__ float sh_cnt[8];

    if (lane == 0) {
        if (warp < 8) {
            sh_mean[warp] = mean;
            sh_m2[warp]   = m2;
            sh_cnt[warp]  = cnt;
        }
    }
    __syncthreads();

    float bmean = 0.0f, bm2 = 0.0f, bcnt = 0.0f;
    if (warp == 0) {
        if (lane < nwarps) {
            bmean = sh_mean[lane];
            bm2   = sh_m2[lane];
            bcnt  = sh_cnt[lane];
        }
        warp_welford_reduce(bmean, bm2, bcnt);
        if (lane == 0) {
            sh_mean[0] = bmean;
            sh_m2[0]   = bm2;
            sh_cnt[0]  = bcnt;
        }
    }
    __syncthreads();

    float final_mean = sh_mean[0];
    float final_m2   = sh_m2[0];
    float final_cnt  = sh_cnt[0];

    float var = (final_cnt > 0.0f) ? (final_m2 / final_cnt) : 0.0f;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < C; c += blockDim.x) {
        float v = xrow[c] + rrow[c];
        float n = (v - final_mean) * inv_std;
        float g = gamma[c];
        float b = beta[c];
        yrow[c] = fmaf(n, g, b);
    }
}

torch::Tensor residual_layernorm_fwd_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
){
    CHECK_CUDA(x); CHECK_CUDA(residual); CHECK_CUDA(gamma); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(residual); CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT(x); CHECK_FLOAT(residual); CHECK_FLOAT(gamma); CHECK_FLOAT(beta);

    TORCH_CHECK(x.dim() == 2 && residual.dim() == 2, "x and residual must be [N, C]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma and beta must be [C]");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    TORCH_CHECK(x.size(1) == gamma.size(0) && x.size(1) == beta.size(0), "gamma/beta must match C");

    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    TORCH_CHECK(N64 <= INT_MAX && C64 <= INT_MAX, "sizes too large");
    int N = (int)N64;
    int C = (int)C64;

    auto y = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (C == 128) {
        // Ensure 16-byte alignment for float4: contiguous float tensor from PyTorch is typically aligned,
        // but we guard in Python by using contiguous() on inputs.
        int threads = 128; // 4 warps per block
        int warps_per_block = threads / 32;
        int blocks = (N + warps_per_block - 1) / warps_per_block;
        residual_layernorm_c128_warp_fwd<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            residual.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N,
            (float)eps
        );
    } else {
        dim3 grid((unsigned)N, 1, 1);
        dim3 block(256, 1, 1);
        residual_layernorm_block_welford_fwd<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            residual.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C,
            (float)eps
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "residual_layernorm_fwd_cuda kernel launch failed: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("residual_layernorm_fwd_cuda", &residual_layernorm_fwd_cuda,
          "Residual + LayerNorm forward (float32, CUDA; C128 warp fastpath)");
}
"""

cpp_src = r"""
torch::Tensor residual_layernorm_fwd_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_vision_attention_resln_c128warp_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=None,  # explicit PYBIND11_MODULE
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        B, C, H, W = x.shape
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor inputs")

        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        seq_len = H * W
        # (S, B, C) contiguous
        x_seq = x.view(B, C, seq_len).permute(2, 0, 1).contiguous()

        attn_out, _ = self.attn(x_seq, x_seq, x_seq)  # (S, B, C)

        # IMPORTANT: keep the exact same (S,B) row mapping for both tensors:
        # flatten (S,B,C) -> (S*B, C) in the same memory order.
        N = seq_len * B
        x2 = attn_out.contiguous().view(N, C)
        r2 = x_seq.contiguous().view(N, C)

        gamma = self.norm.weight
        beta = self.norm.bias
        if gamma.dtype != torch.float32:
            gamma = gamma.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        gamma = gamma.to(device=x.device).contiguous()
        beta = beta.to(device=x.device).contiguous()

        y2 = self.custom_ops.residual_layernorm_fwd_cuda(
            x2, r2, gamma, beta, float(self.norm.eps)
        )

        y_seq = y2.view(seq_len, B, C)
        out = y_seq.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return out