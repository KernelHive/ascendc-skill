import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------- CUDA/C++ extension: BN(training stats) + GELU(exact) + ReLU (faster stats + tiled apply) --------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include <math.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif

static __forceinline__ __device__ float relu_f32(float x) { return x > 0.0f ? x : 0.0f; }

// Exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
static __forceinline__ __device__ float gelu_erf_fwd(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

// Optional fast GELU approximation (tanh-based). Disabled by default to preserve exactness.
// Define USE_GELU_TANH=1 in compile flags to enable.
static __forceinline__ __device__ float gelu_tanh_fwd(float x) {
    // 0.5 x (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float t = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

static __forceinline__ __device__ float gelu_fwd(float x) {
#if defined(USE_GELU_TANH) && (USE_GELU_TANH==1)
    return gelu_tanh_fwd(x);
#else
    return gelu_erf_fwd(x);
#endif
}

static __forceinline__ __device__ float4 load_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
static __forceinline__ __device__ void store_float4(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

static __forceinline__ __device__ float4 ldg_float4(const float* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    float4 out;
    out.x = __ldg(p + 0);
    out.y = __ldg(p + 1);
    out.z = __ldg(p + 2);
    out.w = __ldg(p + 3);
    return out;
#else
    return load_float4(p);
#endif
}

// ---------------- Workspace reuse (per-device) for partial buffers ----------------
struct WSBuf {
    torch::Tensor t;
    int64_t bytes = 0;
};
static WSBuf g_ws_psum[16];
static WSBuf g_ws_psumsq[16];

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
        // over-allocate to reduce realloc frequency
        int64_t alloc_bytes = need_bytes + (need_bytes >> 1);
        int64_t alloc_numel = (alloc_bytes + (int64_t)sizeof(float) - 1) / (int64_t)sizeof(float);
        buf.t = torch::empty({alloc_numel}, opts);
        buf.bytes = alloc_numel * (int64_t)sizeof(float);
    }
    return buf.t.narrow(0, 0, numel);
}

// ---------------- Stage-1: partial sum/sumsq over row partitions (vec4 channels), no atomics ----------------
// Grid:
//   blockIdx.x: channel tile (TILE_C4 float4 channels)
//   blockIdx.y: partition p in [0,P)
template<int BLOCK_THREADS, int TILE_C4>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void bn_stats_stage1_partials_f32x4(
    const float* __restrict__ X, // [M,N]
    float* __restrict__ psum,    // [P,N]
    float* __restrict__ psumsq,  // [P,N]
    int M, int N, int P
) {
    int tile = (int)blockIdx.x;
    int p    = (int)blockIdx.y;

    int N4 = N >> 2;
    int c4_start = tile * TILE_C4;
    if (c4_start >= N4) return;
    int c4_end = c4_start + TILE_C4;
    if (c4_end > N4) c4_end = N4;

    // balanced partitioning of rows
    int m0 = (int)((int64_t)p * (int64_t)M / (int64_t)P);
    int m1 = (int)((int64_t)(p + 1) * (int64_t)M / (int64_t)P);

    for (int c4 = c4_start + (int)threadIdx.x; c4 < c4_end; c4 += BLOCK_THREADS) {
        int c = c4 << 2;

        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        float q0=0.f, q1=0.f, q2=0.f, q3=0.f;

        const float* xptr = X + (int64_t)m0 * (int64_t)N + c;

        #pragma unroll 1
        for (int m = m0; m < m1; ++m) {
            float4 xv = load_float4(xptr);
            xptr += N;
            s0 += xv.x; s1 += xv.y; s2 += xv.z; s3 += xv.w;
            q0 = fmaf(xv.x, xv.x, q0);
            q1 = fmaf(xv.y, xv.y, q1);
            q2 = fmaf(xv.z, xv.z, q2);
            q3 = fmaf(xv.w, xv.w, q3);
        }

        int64_t off = (int64_t)p * (int64_t)N + c;
        psum[off + 0] = s0; psum[off + 1] = s1; psum[off + 2] = s2; psum[off + 3] = s3;
        psumsq[off + 0] = q0; psumsq[off + 1] = q1; psumsq[off + 2] = q2; psumsq[off + 3] = q3;
    }
}

// ---------------- Stage-2: reduce partials -> a,b directly ----------------
// a[c] = gamma[c] * invstd[c]
// b[c] = beta[c] - mean[c] * a[c]
template<int BLOCK_THREADS, int TILE_C4>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void bn_stats_stage2_reduce_ab_f32x4(
    const float* __restrict__ psum,   // [P,N]
    const float* __restrict__ psumsq, // [P,N]
    const float* __restrict__ gamma,  // [N]
    const float* __restrict__ beta,   // [N]
    float* __restrict__ a,            // [N]
    float* __restrict__ b,            // [N]
    int M, int N, int P, float eps
) {
    int tile = (int)blockIdx.x;
    int N4 = N >> 2;
    int c4_start = tile * TILE_C4;
    if (c4_start >= N4) return;
    int c4_end = c4_start + TILE_C4;
    if (c4_end > N4) c4_end = N4;

    for (int c4 = c4_start + (int)threadIdx.x; c4 < c4_end; c4 += BLOCK_THREADS) {
        int c = c4 << 2;

        float s0=0.f,s1=0.f,s2=0.f,s3=0.f;
        float q0=0.f,q1=0.f,q2=0.f,q3=0.f;

        #pragma unroll 1
        for (int p = 0; p < P; ++p) {
            int64_t off = (int64_t)p * (int64_t)N + c;
            s0 += LDG(psum + off + 0);
            s1 += LDG(psum + off + 1);
            s2 += LDG(psum + off + 2);
            s3 += LDG(psum + off + 3);

            q0 += LDG(psumsq + off + 0);
            q1 += LDG(psumsq + off + 1);
            q2 += LDG(psumsq + off + 2);
            q3 += LDG(psumsq + off + 3);
        }

        float invM = 1.0f / (float)M;
        float m0 = s0 * invM, m1 = s1 * invM, m2 = s2 * invM, m3 = s3 * invM;
        float ex20 = q0 * invM, ex21 = q1 * invM, ex22 = q2 * invM, ex23 = q3 * invM;

        float v0 = ex20 - m0*m0;
        float v1 = ex21 - m1*m1;
        float v2 = ex22 - m2*m2;
        float v3 = ex23 - m3*m3;
        v0 = v0 < 0.0f ? 0.0f : v0;
        v1 = v1 < 0.0f ? 0.0f : v1;
        v2 = v2 < 0.0f ? 0.0f : v2;
        v3 = v3 < 0.0f ? 0.0f : v3;

        float inv0 = rsqrtf(v0 + eps);
        float inv1 = rsqrtf(v1 + eps);
        float inv2 = rsqrtf(v2 + eps);
        float inv3 = rsqrtf(v3 + eps);

        float4 g = ldg_float4(gamma + c);
        float4 be = ldg_float4(beta + c);

        float aa0 = g.x * inv0;
        float aa1 = g.y * inv1;
        float aa2 = g.z * inv2;
        float aa3 = g.w * inv3;

        a[c + 0] = aa0; a[c + 1] = aa1; a[c + 2] = aa2; a[c + 3] = aa3;
        b[c + 0] = be.x - m0 * aa0;
        b[c + 1] = be.y - m1 * aa1;
        b[c + 2] = be.z - m2 * aa2;
        b[c + 3] = be.w - m3 * aa3;
    }
}

// ---------------- Stage-3: tiled apply with shared-memory staging of a/b ----------------
// Grid:
//   blockIdx.x: channel tile (TILE_C4 float4 channels)
//   blockIdx.y: row tile (ROW_TILE rows)
// Threads cooperate over channels; for each row in tile, stream X->Y with vec4 IO.
template<int BLOCK_THREADS, int TILE_C4, int ROW_TILE>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void bn_gelu_relu_apply_tiled_smab_f32x4(
    const float* __restrict__ X, // [M,N]
    float* __restrict__ Y,       // [M,N]
    const float* __restrict__ a, // [N]
    const float* __restrict__ b, // [N]
    int M, int N
) {
    int tile = (int)blockIdx.x;
    int row_tile = (int)blockIdx.y;

    int N4 = N >> 2;
    int c4_start = tile * TILE_C4;
    if (c4_start >= N4) return;
    int c4_end = c4_start + TILE_C4;
    if (c4_end > N4) c4_end = N4;
    int c4_count = c4_end - c4_start;

    int m0 = row_tile * ROW_TILE;
    if (m0 >= M) return;

    __shared__ float4 sh_a[TILE_C4];
    __shared__ float4 sh_b[TILE_C4];

    for (int local = (int)threadIdx.x; local < c4_count; local += BLOCK_THREADS) {
        int c = (c4_start + local) << 2;
        sh_a[local] = ldg_float4(a + c);
        sh_b[local] = ldg_float4(b + c);
    }
    __syncthreads();

    #pragma unroll
    for (int rm = 0; rm < ROW_TILE; ++rm) {
        int m = m0 + rm;
        if (m >= M) break;

        const float* xrow = X + (int64_t)m * (int64_t)N;
        float* yrow = Y + (int64_t)m * (int64_t)N;

        for (int local = (int)threadIdx.x; local < c4_count; local += BLOCK_THREADS) {
            int c = (c4_start + local) << 2;
            float4 xv = load_float4(xrow + c);
            float4 av = sh_a[local];
            float4 bv = sh_b[local];

            float y0 = fmaf(xv.x, av.x, bv.x);
            float y1 = fmaf(xv.y, av.y, bv.y);
            float y2 = fmaf(xv.z, av.z, bv.z);
            float y3 = fmaf(xv.w, av.w, bv.w);

            y0 = relu_f32(gelu_fwd(y0));
            y1 = relu_f32(gelu_fwd(y1));
            y2 = relu_f32(gelu_fwd(y2));
            y3 = relu_f32(gelu_fwd(y3));

            store_float4(yrow + c, make_float4(y0, y1, y2, y3));
        }
    }
}

torch::Tensor bn_train_gelu_relu_cuda(
    torch::Tensor X,        // (M,N) float32 cuda contiguous
    torch::Tensor gamma,    // (N) float32 cuda contiguous
    torch::Tensor beta,     // (N) float32 cuda contiguous
    double eps
) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "X/gamma/beta must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32,
                "float32 only");
    TORCH_CHECK(X.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "contiguous only");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");

    int64_t M64 = X.size(0);
    int64_t N64 = X.size(1);
    TORCH_CHECK(gamma.numel() == N64 && beta.numel() == N64, "gamma/beta size mismatch with X.size(1)");
    TORCH_CHECK(M64 > 0 && N64 > 0, "Empty tensors not supported");
    TORCH_CHECK(N64 <= INT32_MAX && M64 <= INT32_MAX, "M/N too large");
    TORCH_CHECK((N64 % 4) == 0, "N must be divisible by 4 for vectorized path");

    int M = (int)M64;
    int N = (int)N64;

    auto Y = torch::empty({M64, N64}, X.options());
    auto a = torch::empty({N64}, X.options());
    auto b = torch::empty({N64}, X.options());

    // Choose P (row partitions) heuristically for this BN size
    int P = 64;
    if (M <= 2048) P = 16;
    else if (M <= 8192) P = 32;
    else P = 64;
    if (P > M) P = M;
    if (P < 1) P = 1;

    // Workspace reuse for partials
    int d = dev_id();
    const auto opts = X.options();
    auto ps  = get_ws_tensor(g_ws_psum[d],   (int64_t)P * (int64_t)N, opts);
    auto psq = get_ws_tensor(g_ws_psumsq[d], (int64_t)P * (int64_t)N, opts);
    float* psum = (float*)ps.data_ptr<float>();
    float* psumsq = (float*)psq.data_ptr<float>();

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Tile config (channels)
    constexpr int TILE_C4 = 64; // 64 float4 = 256 floats
    int N4 = N >> 2;
    int grid_x = (N4 + TILE_C4 - 1) / TILE_C4;

    // Stage-1: partials
    {
        constexpr int BLOCK1 = 256;
        dim3 grid1((unsigned)grid_x, (unsigned)P, 1);
        bn_stats_stage1_partials_f32x4<BLOCK1, TILE_C4><<<grid1, BLOCK1, 0, stream>>>(
            (const float*)X.data_ptr<float>(),
            psum,
            psumsq,
            M, N, P
        );
    }

    // Stage-2: reduce -> a,b (fused finalize)
    {
        constexpr int BLOCK2 = 128; // smaller to reduce register pressure
        dim3 grid2((unsigned)grid_x, 1, 1);
        bn_stats_stage2_reduce_ab_f32x4<BLOCK2, TILE_C4><<<grid2, BLOCK2, 0, stream>>>(
            (const float*)psum,
            (const float*)psumsq,
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (float*)a.data_ptr<float>(),
            (float*)b.data_ptr<float>(),
            M, N, P, (float)eps
        );
    }

    // Stage-3: apply tiled with SM staging of a/b
    {
        constexpr int BLOCK3 = 256;
        constexpr int ROW_TILE = 8;
        int grid_y = (M + ROW_TILE - 1) / ROW_TILE;
        dim3 grid3((unsigned)grid_x, (unsigned)grid_y, 1);
        bn_gelu_relu_apply_tiled_smab_f32x4<BLOCK3, TILE_C4, ROW_TILE><<<grid3, BLOCK3, 0, stream>>>(
            (const float*)X.data_ptr<float>(),
            (float*)Y.data_ptr<float>(),
            (const float*)a.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            M, N
        );
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bn_train_gelu_relu_cuda", &bn_train_gelu_relu_cuda,
          "BatchNorm1d(training stats) + GELU + ReLU (CUDA; 2-stage stats, tiled apply)");
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor bn_train_gelu_relu_cuda(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_bn_train_gelu_relu_v4_partials_tiled",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        # To enable tanh-approx GELU (faster, slightly different numerics), uncomment:
        # "-DUSE_GELU_TANH=1",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)

# -------------------- Optimized model wrapper --------------------

class ModelNew(nn.Module):
    """
    Optimized model:
      Linear (cuBLAS) -> fused BN(training stats, no running stat update) + GELU + ReLU (custom CUDA)

    Notes:
      - Forward output matches the "training batch-statistics" path of BatchNorm1d for affine params (weight/bias).
      - Running stats are not updated in the custom op (behavioral difference vs nn.BatchNorm1d in train()).
      - Requires CUDA float32 contiguous and out_features % 4 == 0.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=True)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.custom_ops_lib = custom_ops_lib

    @staticmethod
    def _to_cuda_f32_contig(t: torch.Tensor, device: torch.device) -> torch.Tensor:
        if (not t.is_cuda) or (t.device != device) or (t.dtype != torch.float32):
            t = t.to(device=device, dtype=torch.float32)
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        y = self.gemm(x)

        bn = self.batch_norm
        dev = y.device
        gamma = self._to_cuda_f32_contig(bn.weight, dev)
        beta = self._to_cuda_f32_contig(bn.bias, dev)
        eps = float(bn.eps)

        if not y.is_contiguous():
            y = y.contiguous()

        if (y.dim() != 2) or ((y.size(1) & 3) != 0):
            y = bn(y)
            y = torch.nn.functional.gelu(y)
            y = torch.relu(y)
            return y

        return self.custom_ops_lib.bn_train_gelu_relu_cuda(y, gamma, beta, eps)


# Compatibility helpers
batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]