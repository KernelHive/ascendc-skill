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

// ---------------- Workspace reuse for partial buffers ----------------
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
    if (!buf.t.defined() || buf.bytes < need_bytes || buf.t.device() != opts.device()) {
        int64_t alloc_bytes = need_bytes + (need_bytes >> 1);
        int64_t alloc_numel = (alloc_bytes + (int64_t)sizeof(float) - 1) / (int64_t)sizeof(float);
        buf.t = torch::empty({alloc_numel}, opts);
        buf.bytes = alloc_numel * (int64_t)sizeof(float);
    }
    return buf.t.narrow(0, 0, numel);
}

// ---------------- Stage-1: partial sum/sumsq over B slices (vec4 channels) ----------------
template<int BLOCK_THREADS, int TILE_C4>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void stats_stage1_kernel(
    const float* __restrict__ X,     // [B,C]
    const float* __restrict__ scale, // [C]
    float* __restrict__ partial_sum,   // [P,C]
    float* __restrict__ partial_sumsq, // [P,C]
    int B, int C, int P
) {
    int tile = (int)blockIdx.x;
    int p    = (int)blockIdx.y;

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

        const float* xptr = X + (size_t)b0 * (size_t)C + (size_t)c;
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

        size_t off = (size_t)p * (size_t)C + (size_t)c;
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

// ---------------- Stage-2: reduce partials -> mean + inv_std (vec4 channels) ----------------
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

        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        float q0=0.f, q1=0.f, q2=0.f, q3=0.f;

        #pragma unroll 1
        for (int p = 0; p < P; ++p) {
            size_t off = (size_t)p * (size_t)C + (size_t)c;
            s0 += ldg_f32(partial_sum + off + 0);
            s1 += ldg_f32(partial_sum + off + 1);
            s2 += ldg_f32(partial_sum + off + 2);
            s3 += ldg_f32(partial_sum + off + 3);

            q0 += ldg_f32(partial_sumsq + off + 0);
            q1 += ldg_f32(partial_sumsq + off + 1);
            q2 += ldg_f32(partial_sumsq + off + 2);
            q3 += ldg_f32(partial_sumsq + off + 3);
        }

        float invB = 1.0f / (float)B;
        float m0 = s0 * invB;
        float m1 = s1 * invB;
        float m2 = s2 * invB;
        float m3 = s3 * invB;

        float ex20 = q0 * invB;
        float ex21 = q1 * invB;
        float ex22 = q2 * invB;
        float ex23 = q3 * invB;

        float v0 = ex20 - m0 * m0;
        float v1 = ex21 - m1 * m1;
        float v2 = ex22 - m2 * m2;
        float v3 = ex23 - m3 * m3;

        v0 = v0 < 0.0f ? 0.0f : v0;
        v1 = v1 < 0.0f ? 0.0f : v1;
        v2 = v2 < 0.0f ? 0.0f : v2;
        v3 = v3 < 0.0f ? 0.0f : v3;

        mean[c + 0] = m0;
        mean[c + 1] = m1;
        mean[c + 2] = m2;
        mean[c + 3] = m3;

        invstd[c + 0] = rsqrtf(v0 + eps);
        invstd[c + 1] = rsqrtf(v1 + eps);
        invstd[c + 2] = rsqrtf(v2 + eps);
        invstd[c + 3] = rsqrtf(v3 + eps);
    }
}

// ---------------- Stage-3 (optimized): tiled apply with shared-memory parameter staging ----------------
// Grid:
//   blockIdx.x: channel tile (TILE_C4 float4 channels)
//   blockIdx.y: row tile (ROW_TILE rows)
// Threads cooperate over c4 within tile; for each row in row tile they read X and write Out contiguously.
// Params are loaded once per tile into shared memory (scale, mean, invstd, gamma, beta).
template<int BLOCK_THREADS, int TILE_C4, int ROW_TILE>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void apply_kernel_tiled_smparams(
    const float* __restrict__ X,     // [B,C]
    const float* __restrict__ scale, // [C]
    const float* __restrict__ mean,  // [C]
    const float* __restrict__ invstd,// [C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ Out,         // [B,C]
    int B, int C
) {
    int tile = (int)blockIdx.x;
    int row_tile = (int)blockIdx.y;

    int C4 = C >> 2;
    int c4_start = tile * TILE_C4;
    if (c4_start >= C4) return;
    int c4_end = c4_start + TILE_C4;
    if (c4_end > C4) c4_end = C4;
    int c4_count = c4_end - c4_start;

    int b0 = row_tile * ROW_TILE;
    if (b0 >= B) return;

    // shared param tiles (float4)
    __shared__ float4 sh_scale[TILE_C4];
    __shared__ float4 sh_mean [TILE_C4];
    __shared__ float4 sh_inv  [TILE_C4];
    __shared__ float4 sh_gamma[TILE_C4];
    __shared__ float4 sh_beta [TILE_C4];

    // Stage params: one float4 per "local c4"
    for (int local = (int)threadIdx.x; local < c4_count; local += BLOCK_THREADS) {
        int c = (c4_start + local) << 2;
        sh_scale[local] = ldg_f32x4(reinterpret_cast<const float4*>(scale + c));
        sh_mean [local] = ldg_f32x4(reinterpret_cast<const float4*>(mean  + c));
        sh_inv  [local] = ldg_f32x4(reinterpret_cast<const float4*>(invstd+ c));
        sh_gamma[local] = ldg_f32x4(reinterpret_cast<const float4*>(gamma + c));
        sh_beta [local] = ldg_f32x4(reinterpret_cast<const float4*>(beta  + c));
    }
    __syncthreads();

    // Apply for ROW_TILE rows
    #pragma unroll
    for (int rb = 0; rb < ROW_TILE; ++rb) {
        int b = b0 + rb;
        if (b >= B) break;

        const float* xrow = X + (size_t)b * (size_t)C;
        float* orow = Out + (size_t)b * (size_t)C;

        for (int local = (int)threadIdx.x; local < c4_count; local += BLOCK_THREADS) {
            int c = (c4_start + local) << 2;

            float4 xv = *reinterpret_cast<const float4*>(xrow + c);

            float4 sc = sh_scale[local];
            float4 m  = sh_mean [local];
            float4 inv= sh_inv  [local];
            float4 g  = sh_gamma[local];
            float4 be = sh_beta [local];

            float y0 = (xv.x * sc.x - m.x) * inv.x;
            float y1 = (xv.y * sc.y - m.y) * inv.y;
            float y2 = (xv.z * sc.z - m.z) * inv.z;
            float y3 = (xv.w * sc.w - m.w) * inv.w;

            float4 ov;
            ov.x = fmaf(y0, g.x, be.x);
            ov.y = fmaf(y1, g.y, be.y);
            ov.z = fmaf(y2, g.z, be.z);
            ov.w = fmaf(y3, g.w, be.w);

            *reinterpret_cast<float4*>(orow + c) = ov;
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

    // Workspace reuse
    const auto opts = x.options();
    int d = dev_id();
    auto ps  = get_ws_tensor(g_ws_sum[d],   (int64_t)P * (int64_t)C, opts);
    auto psq = get_ws_tensor(g_ws_sumsq[d], (int64_t)P * (int64_t)C, opts);

    float* partial_sum   = (float*)ps.data_ptr<float>();
    float* partial_sumsq = (float*)psq.data_ptr<float>();

    // Launch config
    constexpr int TILE_C4 = 64; // 64 float4 = 256 floats per tile (good coalescing)
    int C4 = C >> 2;
    int grid_x = (C4 + TILE_C4 - 1) / TILE_C4;

    // Stage-1
    constexpr int BLOCK1 = 256;
    dim3 grid1((unsigned)grid_x, (unsigned)P);
    stats_stage1_kernel<BLOCK1, TILE_C4><<<grid1, BLOCK1>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)scale.data_ptr<float>(),
        partial_sum,
        partial_sumsq,
        B, C, P
    );

    // Stage-2 (smaller CTA to reduce register pressure; stage-2 is not the dominant bandwidth kernel)
    constexpr int BLOCK2 = 128;
    dim3 grid2((unsigned)grid_x);
    stats_stage2_kernel<BLOCK2, TILE_C4><<<grid2, BLOCK2>>>(
        (const float*)partial_sum,
        (const float*)partial_sumsq,
        (float*)mean.data_ptr<float>(),
        (float*)invsd.data_ptr<float>(),
        B, C, P, (float)eps
    );

    // Stage-3 (dominant): tiled apply with SM parameter staging and 2D grid
    constexpr int BLOCK3 = 256;
    constexpr int ROW_TILE = 8;  // amortize param staging; keep CTA short enough for good scheduling
    int grid_y = (B + ROW_TILE - 1) / ROW_TILE;
    dim3 grid3((unsigned)grid_x, (unsigned)grid_y);
    apply_kernel_tiled_smparams<BLOCK3, TILE_C4, ROW_TILE><<<grid3, BLOCK3>>>(
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
    name="custom_ops_lib_scale_bn_stage12_tiled_apply_smparams_ws",
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
      - eval() mode (uses running stats)
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

        return self.custom_ops_lib.scale_batch_norm_train_cuda(
            y, scale, gamma, beta, float(self._bn_eps)
        )