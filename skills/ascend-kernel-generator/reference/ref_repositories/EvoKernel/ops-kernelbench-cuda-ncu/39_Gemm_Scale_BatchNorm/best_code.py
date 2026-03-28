import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FP32
#define CHECK_FP32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ float4 ldg_f32x4(const float4* p) {
#if __CUDA_ARCH__ >= 350
    float4 out;
    const float* pf = reinterpret_cast<const float*>(p);
    out.x = __ldg(pf + 0);
    out.y = __ldg(pf + 1);
    out.z = __ldg(pf + 2);
    out.w = __ldg(pf + 3);
    return out;
#else
    return *p;
#endif
}

// ---------------- Workspace reuse ----------------
struct WSBuf {
    torch::Tensor t;
    int64_t bytes = 0;
};
static WSBuf g_ws_sum[16];
static WSBuf g_ws_sumsq[16];

static inline int dev_id() {
    int d = 0;
    cudaGetDevice(&d);
    if (d < 0) d = 0;
    if (d > 15) d = 15;
    return d;
}

static torch::Tensor get_ws_tensor(WSBuf& buf, int64_t numel, const torch::TensorOptions& opts) {
    int64_t need_bytes = numel * (int64_t)sizeof(float);
    if (!buf.t.defined() || buf.bytes < need_bytes) {
        // grow ~1.5x
        int64_t alloc_bytes = need_bytes + (need_bytes >> 1);
        int64_t alloc_numel = (alloc_bytes + (int64_t)sizeof(float) - 1) / (int64_t)sizeof(float);
        buf.t = torch::empty({alloc_numel}, opts);
        buf.bytes = alloc_numel * (int64_t)sizeof(float);
    }
    return buf.t.narrow(0, 0, numel);
}

// ---------------- Stage-1: partial sums/sumsq over B slices ----------------
template<int BLOCK_THREADS, int TILE_C4>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void stats_stage1_kernel(
    const float* __restrict__ X,      // [B,C]
    const float* __restrict__ scale,  // [C]
    float* __restrict__ partial_sum,  // [P,C]
    float* __restrict__ partial_sumsq,// [P,C]
    int B, int C, int P
) {
    int tile = (int)blockIdx.x;   // tile over C4
    int p    = (int)blockIdx.y;   // partial id
    int C4 = C >> 2;
    int c4_start = tile * TILE_C4;
    if (c4_start >= C4) return;
    int c4_end = c4_start + TILE_C4;
    if (c4_end > C4) c4_end = C4;

    int b0 = (int)((int64_t)p * B / P);
    int b1 = (int)((int64_t)(p + 1) * B / P);

    for (int c4 = c4_start + (int)threadIdx.x; c4 < c4_end; c4 += BLOCK_THREADS) {
        int c = c4 << 2;
        float4 sc = ldg_f32x4(reinterpret_cast<const float4*>(scale + c));

        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        float q0=0.f, q1=0.f, q2=0.f, q3=0.f;

        const float* xptr = X + (size_t)b0 * C + c;
        #pragma unroll 1
        for (int b = b0; b < b1; ++b) {
            float4 xv = *reinterpret_cast<const float4*>(xptr);
            xptr += C;
            float v0 = xv.x * sc.x;
            float v1 = xv.y * sc.y;
            float v2 = xv.z * sc.z;
            float v3 = xv.w * sc.w;
            s0 += v0; s1 += v1; s2 += v2; s3 += v3;
            q0 = fmaf(v0, v0, q0);
            q1 = fmaf(v1, v1, q1);
            q2 = fmaf(v2, v2, q2);
            q3 = fmaf(v3, v3, q3);
        }

        size_t off = (size_t)p * C + c;
        partial_sum[off + 0] = s0;
        partial_sum[off + 1] = s1;
        partial_sum[off + 2] = s2;
        partial_sum[off + 3] = s3;

        partial_sumsq[off + 0] = q0;
        partial_sumsq[off + 1] = q1;
        partial_sumsq[off + 2] = q2;
        partial_sumsq[off + 3] = q3;
    }
}

// ---------------- Stage-2: reduce partials -> mean + invstd per channel ----------------
template<int BLOCK_THREADS, int TILE_C4>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void stats_stage2_kernel(
    const float* __restrict__ partial_sum,   // [P,C]
    const float* __restrict__ partial_sumsq, // [P,C]
    float* __restrict__ mean,    // [C]
    float* __restrict__ invstd,  // [C]
    int B, int C, int P, float eps
) {
    int tile = (int)blockIdx.x;
    int C4 = C >> 2;
    int c4_start = tile * TILE_C4;
    if (c4_start >= C4) return;
    int c4_end = c4_start + TILE_C4;
    if (c4_end > C4) c4_end = C4;

    for (int c4 = c4_start + (int)threadIdx.x; c4 < c4_end; c4 += BLOCK_THREADS) {
        int c = c4 << 2;

        float4 s = {0.f,0.f,0.f,0.f};
        float4 q = {0.f,0.f,0.f,0.f};

        // vectorized loads over P
        #pragma unroll 1
        for (int p = 0; p < P; ++p) {
            size_t off = (size_t)p * C + c;
            float4 ps = ldg_f32x4(reinterpret_cast<const float4*>(partial_sum + off));
            float4 pq = ldg_f32x4(reinterpret_cast<const float4*>(partial_sumsq + off));
            s.x += ps.x; s.y += ps.y; s.z += ps.z; s.w += ps.w;
            q.x += pq.x; q.y += pq.y; q.z += pq.z; q.w += pq.w;
        }

        float invB = 1.0f / (float)B;
        float4 m;
        m.x = s.x * invB; m.y = s.y * invB; m.z = s.z * invB; m.w = s.w * invB;

        float4 ex2;
        ex2.x = q.x * invB; ex2.y = q.y * invB; ex2.z = q.z * invB; ex2.w = q.w * invB;

        float v0 = ex2.x - m.x * m.x;
        float v1 = ex2.y - m.y * m.y;
        float v2 = ex2.z - m.z * m.z;
        float v3 = ex2.w - m.w * m.w;
        v0 = v0 < 0.f ? 0.f : v0;
        v1 = v1 < 0.f ? 0.f : v1;
        v2 = v2 < 0.f ? 0.f : v2;
        v3 = v3 < 0.f ? 0.f : v3;

        mean[c + 0] = m.x; mean[c + 1] = m.y; mean[c + 2] = m.z; mean[c + 3] = m.w;
        invstd[c + 0] = rsqrtf(v0 + eps);
        invstd[c + 1] = rsqrtf(v1 + eps);
        invstd[c + 2] = rsqrtf(v2 + eps);
        invstd[c + 3] = rsqrtf(v3 + eps);
    }
}

// ---------------- Apply: 2D tiled, shared-memory parameter staging ----------------
// Grid:
//   blockIdx.x: channel tile in float4 units (TILE_C4_APPLY)
//   blockIdx.y: row tile (ROWS_PER_BLOCK)
// Threads cover channels (float4) within tile; each thread processes VEC_PER_THREAD float4s.
template<int BLOCK_THREADS, int TILE_C4_APPLY, int ROWS_PER_BLOCK, int VEC_PER_THREAD>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void apply_2d_tile_kernel(
    const float* __restrict__ X,     // [B,C]
    const float* __restrict__ scale, // [C]
    const float* __restrict__ mean,  // [C]
    const float* __restrict__ invstd,// [C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ Out,         // [B,C]
    int B, int C
) {
    int C4 = C >> 2;
    int tile_c4 = (int)blockIdx.x * TILE_C4_APPLY;
    int row0 = (int)blockIdx.y * ROWS_PER_BLOCK;
    if (tile_c4 >= C4 || row0 >= B) return;

    int tile_end = tile_c4 + TILE_C4_APPLY;
    if (tile_end > C4) tile_end = C4;
    int tile_count = tile_end - tile_c4;

    // Shared staging of parameters for this channel tile.
    // Keep as float4 arrays; total shmem = 5*TILE_C4_APPLY*sizeof(float4).
    extern __shared__ float4 sh4[];
    float4* sh_scale = sh4 + 0 * TILE_C4_APPLY;
    float4* sh_mean  = sh4 + 1 * TILE_C4_APPLY;
    float4* sh_inv   = sh4 + 2 * TILE_C4_APPLY;
    float4* sh_gamma = sh4 + 3 * TILE_C4_APPLY;
    float4* sh_beta  = sh4 + 4 * TILE_C4_APPLY;

    // Cooperative load (vectorized)
    for (int i = (int)threadIdx.x; i < tile_count; i += BLOCK_THREADS) {
        int c4 = tile_c4 + i;
        int c = c4 << 2;
        sh_scale[i] = ldg_f32x4(reinterpret_cast<const float4*>(scale + c));
        sh_mean[i]  = ldg_f32x4(reinterpret_cast<const float4*>(mean  + c));
        sh_inv[i]   = ldg_f32x4(reinterpret_cast<const float4*>(invstd+ c));
        sh_gamma[i] = ldg_f32x4(reinterpret_cast<const float4*>(gamma + c));
        sh_beta[i]  = ldg_f32x4(reinterpret_cast<const float4*>(beta  + c));
    }
    __syncthreads();

    // Each thread processes VEC_PER_THREAD float4s in channel dimension for ILP.
    int base_i = (int)threadIdx.x * VEC_PER_THREAD;

    #pragma unroll
    for (int rb = 0; rb < ROWS_PER_BLOCK; ++rb) {
        int b = row0 + rb;
        if (b >= B) break;

        size_t row_off = (size_t)b * (size_t)C;

        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int i = base_i + u;
            if (i >= tile_count) break;

            int c = (tile_c4 + i) << 2;

            float4 xv = *reinterpret_cast<const float4*>(X + row_off + c);

            float4 sc = sh_scale[i];
            float4 m  = sh_mean[i];
            float4 inv= sh_inv[i];
            float4 g  = sh_gamma[i];
            float4 be = sh_beta[i];

            float y0 = (xv.x * sc.x - m.x) * inv.x;
            float y1 = (xv.y * sc.y - m.y) * inv.y;
            float y2 = (xv.z * sc.z - m.z) * inv.z;
            float y3 = (xv.w * sc.w - m.w) * inv.w;

            float4 ov;
            ov.x = fmaf(y0, g.x, be.x);
            ov.y = fmaf(y1, g.y, be.y);
            ov.z = fmaf(y2, g.z, be.z);
            ov.w = fmaf(y3, g.w, be.w);

            *reinterpret_cast<float4*>(Out + row_off + c) = ov;
        }
    }
}

torch::Tensor scale_batch_norm_train_cuda(
    torch::Tensor x,        // [B,C]
    torch::Tensor scale,    // [C]
    torch::Tensor gamma,    // [C]
    torch::Tensor beta,     // [C]
    double eps
) {
    CHECK_CUDA(x);
    CHECK_CUDA(scale);
    CHECK_CUDA(gamma);
    CHECK_CUDA(beta);
    CHECK_FP32(x);
    CHECK_FP32(scale);
    CHECK_FP32(gamma);
    CHECK_FP32(beta);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(scale);
    CHECK_CONTIGUOUS(gamma);
    CHECK_CONTIGUOUS(beta);

    TORCH_CHECK(x.dim() == 2, "x must be 2D (B,C)");
    TORCH_CHECK(scale.dim() == 1 && gamma.dim() == 1 && beta.dim() == 1, "params must be 1D");
    int B = (int)x.size(0);
    int C = (int)x.size(1);
    TORCH_CHECK(scale.size(0) == C && gamma.size(0) == C && beta.size(0) == C, "param size mismatch");
    TORCH_CHECK((C & 3) == 0, "Optimized CUDA path requires C % 4 == 0");

    auto out   = torch::empty({B, C}, x.options());
    auto mean  = torch::empty({C}, x.options());
    auto invsd = torch::empty({C}, x.options());

    // Partial count heuristic
    int P = 32;
    if (B <= 256) P = 8;
    else if (B <= 1024) P = 16;
    else if (B <= 4096) P = 32;
    else P = 64;
    if (P > B) P = B;
    if (P < 1) P = 1;

    // Reuse workspace for partials
    const auto opts = x.options();
    int d = dev_id();
    auto ps  = get_ws_tensor(g_ws_sum[d],   (int64_t)P * (int64_t)C, opts);
    auto psq = get_ws_tensor(g_ws_sumsq[d], (int64_t)P * (int64_t)C, opts);
    float* partial_sum   = (float*)ps.data_ptr<float>();
    float* partial_sumsq = (float*)psq.data_ptr<float>();

    // Stage-1
    constexpr int BLOCK1 = 256;
    constexpr int TILE_C4 = 64;
    int C4 = C >> 2;
    int grid_x = (C4 + TILE_C4 - 1) / TILE_C4;
    dim3 grid1((unsigned)grid_x, (unsigned)P);
    stats_stage1_kernel<BLOCK1, TILE_C4><<<grid1, BLOCK1>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)scale.data_ptr<float>(),
        partial_sum,
        partial_sumsq,
        B, C, P
    );

    // Stage-2 (smaller CTA to reduce regs and allow more residency)
    constexpr int BLOCK2 = 128;
    dim3 grid2((unsigned)grid_x);
    stats_stage2_kernel<BLOCK2, TILE_C4><<<grid2, BLOCK2>>>(
        (const float*)partial_sum,
        (const float*)partial_sumsq,
        (float*)mean.data_ptr<float>(),
        (float*)invsd.data_ptr<float>(),
        B, C, P, (float)eps
    );

    // Apply (2D tiled)
    // Choose tile sizes to balance sync overhead vs work per block.
    // For C=4096 -> C4=1024; TILE_C4_APPLY=128 gives 8 tiles in x.
    // ROWS_PER_BLOCK=4 gives many CTAs in y for B=16384 -> 4096 blocks in y.
    constexpr int TILE_C4_APPLY = 128;
    constexpr int ROWS_PER_BLOCK = 4;
    constexpr int BLOCKA = 128;
    constexpr int VEC_PER_THREAD = 2; // each thread covers 2 float4s (8 floats) for ILP

    int grid_ax = (C4 + TILE_C4_APPLY - 1) / TILE_C4_APPLY;
    int grid_ay = (B  + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    size_t shmem_bytes = (size_t)(5 * TILE_C4_APPLY) * sizeof(float4);
    apply_2d_tile_kernel<BLOCKA, TILE_C4_APPLY, ROWS_PER_BLOCK, VEC_PER_THREAD><<< dim3((unsigned)grid_ax, (unsigned)grid_ay), BLOCKA, shmem_bytes >>>(
        (const float*)x.data_ptr<float>(),
        (const float*)scale.data_ptr<float>(),
        (const float*)mean.data_ptr<float>(),
        (const float*)invsd.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, C
    );

    return out;
}
"""

cpp_source = r"""
torch::Tensor scale_batch_norm_train_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_scale_bn_stage12_apply2d_wsreuse_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["scale_batch_norm_train_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps GEMM (Linear) in PyTorch/cuBLAS and fuses:
      mul(scale) + BatchNorm1d (training batch stats; no running stat update)
    into a custom CUDA path.

    Fallback:
      - CPU
      - non-fp32
      - non-contiguous
      - eval() mode
      - C % 4 != 0
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._bn_eps = float(eps)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gemm(x)

        if (not y.is_cuda) or (y.dtype != torch.float32) or (not y.is_contiguous()) or (not self.training):
            y = y * self.scale
            y = self.bn(y)
            return y

        B, C = y.shape
        if (C % 4) != 0:
            y = y * self.scale
            y = self.bn(y)
            return y

        scale = self.scale
        gamma = self.bn.weight
        beta = self.bn.bias
        if gamma is None or beta is None:
            y = y * self.scale
            y = self.bn(y)
            return y

        # Ensure fp32/contiguous params
        if scale.dtype != torch.float32:
            scale = scale.float()
        if gamma.dtype != torch.float32:
            gamma = gamma.float()
        if beta.dtype != torch.float32:
            beta = beta.float()

        if not scale.is_contiguous():
            scale = scale.contiguous()
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()

        return self.custom_ops_lib.scale_batch_norm_train_cuda(y, scale, gamma, beta, float(self._bn_eps))