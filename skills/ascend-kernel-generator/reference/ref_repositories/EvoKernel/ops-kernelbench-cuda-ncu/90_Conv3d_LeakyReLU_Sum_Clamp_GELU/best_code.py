import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA post-ops for Conv3d output:
#   LeakyReLU -> Add(sum_tensor[C]) -> Clamp -> GELU
#
# Incremental improvements vs current baseline:
#  - Keep constant-memory broadcast (fast for small C; avoids failed gmem add path).
#  - Faster channel index reconstruction:
#      * specialize C==64 via bitmask
#      * otherwise use fast divmod-by-constant (mul-hi) to compute (t % C)
#  - Improve scheduling: grid size capped to SM*CTAs_per_SM (avoid huge grids) but still large enough.
#  - Keep vectorized float4 path with modest ILP=2; add tail scalar kernel for leftovers.
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// Constant memory for add vector (up to 4096 floats = 16KB)
__device__ __constant__ float ADD_C_CONST[4096];

__device__ __forceinline__ float leaky_relu_f32(float x, float neg_slope) {
    return x >= 0.0f ? x : x * neg_slope;
}

__device__ __forceinline__ float clamp_f32(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

// Tanh-based GELU approximation (fast; uses fast-math tanhf)
__device__ __forceinline__ float gelu_tanh_f32(float x) {
    // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float t = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float postops_f32(float v, float addv, float neg_slope, float clamp_lo, float clamp_hi) {
    v = leaky_relu_f32(v, neg_slope);
    v = v + addv;
    v = clamp_f32(v, clamp_lo, clamp_hi);
    v = gelu_tanh_f32(v);
    return v;
}

__device__ __forceinline__ float4 postops_f32x4(float4 v, float addv, float neg_slope, float clamp_lo, float clamp_hi) {
    v.x = postops_f32(v.x, addv, neg_slope, clamp_lo, clamp_hi);
    v.y = postops_f32(v.y, addv, neg_slope, clamp_lo, clamp_hi);
    v.z = postops_f32(v.z, addv, neg_slope, clamp_lo, clamp_hi);
    v.w = postops_f32(v.w, addv, neg_slope, clamp_lo, clamp_hi);
    return v;
}

template <bool USE_CONST_ADD>
__device__ __forceinline__ float load_add(int c, const float* __restrict__ add_gmem) {
    if constexpr (USE_CONST_ADD) {
        return ADD_C_CONST[c];
    } else {
#if __CUDA_ARCH__ >= 350
        return __ldg(add_gmem + c);
#else
        return add_gmem[c];
#endif
    }
}

// -----------------------------
// Fast divmod by constant (for t % C).
// We need only remainder. For small C this is cheap.
// -----------------------------
struct FastDivmodU32 {
    uint32_t d;
    uint32_t m; // multiplier for approximate division
    uint32_t s; // shift

    __host__ static FastDivmodU32 make(uint32_t divisor) {
        FastDivmodU32 f{};
        f.d = divisor;
        // Based on Hacker's Delight / libdivide-like approach for unsigned 32-bit.
        // Compute m = floor(2^(32+s)/d), choose s so that works for all n.
        // We'll use a simple method:
        // For d in [1..2^31], let s = 0 and m = floor(2^32 / d) + 1
        // Then q = mulhi(n, m) gives floor(n/d) for many d; it can be off by 1.
        // We'll correct remainder by a single adjustment.
        // This keeps code small; perf benefit comes mainly from avoiding actual div.
        f.s = 0;
        uint64_t one = 1ull << 32;
        f.m = (uint32_t)(one / divisor + 1ull);
        return f;
    }

    __device__ __forceinline__ uint32_t div(uint32_t n) const {
        uint32_t q = __umulhi(n, m);
        // Correct if q*d > n
        uint32_t prod = q * d;
        if (prod > n) q -= 1;
        // Also correct if (q+1)*d <= n
        // (rare with +1 multiplier, but keep safe)
        if ((uint64_t)(q + 1) * (uint64_t)d <= (uint64_t)n) q += 1;
        return q;
    }

    __device__ __forceinline__ uint32_t mod(uint32_t n) const {
        uint32_t q = div(n);
        return n - q * d;
    }
};

// For our sizes, t fits in 32-bit: t = (j/spatial) where j < N*C*spatial.
// We still use 64-bit for j, but cast t to u32 safely (guarded on host).
template <bool USE_CONST_ADD, bool NT_STORE>
__global__ __launch_bounds__(256, 2) void fused_flatten_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_c, // gmem fallback
    float* __restrict__ out,
    int64_t total_elems,
    int C,
    int64_t spatial,
    FastDivmodU32 fdmC,
    float neg_slope,
    float clamp_lo,
    float clamp_hi
) {
    const int tid = threadIdx.x;
    int64_t idx4 = ((int64_t)blockIdx.x * (int64_t)blockDim.x + tid);
    int64_t stride4 = (int64_t)blockDim.x * (int64_t)gridDim.x;

    int64_t total4 = total_elems >> 2;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

    for (int64_t j4 = idx4; j4 < total4; j4 += stride4 * 2) {
#pragma unroll 2
        for (int u = 0; u < 2; ++u) {
            int64_t k4 = j4 + (int64_t)u * stride4;
            if (k4 >= total4) break;

            int64_t base = k4 << 2;           // element index in floats
            int64_t t64  = base / spatial;    // n*C + c

            uint32_t t = (uint32_t)t64;
            int c;
            if (C == 64) {
                c = (int)(t & 63u);
            } else {
                c = (int)fdmC.mod(t);
            }

            float addv = load_add<USE_CONST_ADD>(c, add_c);

#if __CUDA_ARCH__ >= 350
            float4 v = __ldg(x4 + k4);
#else
            float4 v = x4[k4];
#endif
            v = postops_f32x4(v, addv, neg_slope, clamp_lo, clamp_hi);

            if constexpr (NT_STORE) {
#if __CUDA_ARCH__ >= 700
                // Write-through hint to reduce cache pollution when output is not reread soon.
                asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                             :
                             : "l"(o4 + k4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#else
                o4[k4] = v;
#endif
            } else {
                o4[k4] = v;
            }
        }
    }
}

template <bool USE_CONST_ADD>
__global__ __launch_bounds__(256, 2) void fused_tail_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_c,
    float* __restrict__ out,
    int64_t start, // inclusive
    int64_t total_elems,
    int C,
    int64_t spatial,
    FastDivmodU32 fdmC,
    float neg_slope,
    float clamp_lo,
    float clamp_hi
) {
    int64_t idx = start + (int64_t)blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t j = idx; j < total_elems; j += stride) {
        int64_t t64 = j / spatial;
        uint32_t t = (uint32_t)t64;
        int c;
        if (C == 64) c = (int)(t & 63u);
        else c = (int)fdmC.mod(t);

        float addv = load_add<USE_CONST_ADD>(c, add_c);
#if __CUDA_ARCH__ >= 350
        float v = __ldg(x + j);
#else
        float v = x[j];
#endif
        out[j] = postops_f32(v, addv, neg_slope, clamp_lo, clamp_hi);
    }
}

torch::Tensor fused_conv3d_postops_cuda(torch::Tensor x, torch::Tensor add_tensor, double neg_slope, double clamp_lo, double clamp_hi) {
    TORCH_CHECK(x.is_cuda(), "fused_conv3d_postops_cuda: x must be a CUDA tensor");
    TORCH_CHECK(add_tensor.is_cuda(), "fused_conv3d_postops_cuda: add_tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "fused_conv3d_postops_cuda: only float32 supported");
    TORCH_CHECK(add_tensor.scalar_type() == torch::kFloat32, "fused_conv3d_postops_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "fused_conv3d_postops_cuda: x must be contiguous (NCDHW)");
    TORCH_CHECK(add_tensor.is_contiguous(), "fused_conv3d_postops_cuda: add_tensor must be contiguous");
    TORCH_CHECK(x.dim() == 5, "fused_conv3d_postops_cuda: x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(add_tensor.dim() == 4, "fused_conv3d_postops_cuda: add_tensor must be 4D [C,1,1,1]");

    int64_t N = x.size(0);
    int64_t C64 = x.size(1);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    TORCH_CHECK(add_tensor.size(0) == C64, "fused_conv3d_postops_cuda: add_tensor.size(0) must match x.size(1)");
    TORCH_CHECK(add_tensor.size(1) == 1 && add_tensor.size(2) == 1 && add_tensor.size(3) == 1,
                "fused_conv3d_postops_cuda: add_tensor must have shape [C,1,1,1]");

    // Ensure t=(j/spatial) fits in u32 (needed for FastDivmodU32).
    // Here t < N*C; for typical conv outputs it's easily < 2^32.
    TORCH_CHECK((uint64_t)(N * C64) < (1ull<<32), "fused_conv3d_postops_cuda: N*C too large for fast divmod path");

    int C = (int)C64;
    int64_t spatial = D * H * W;
    int64_t total_elems = N * C64 * spatial;

    auto add_c = add_tensor.view({C64});
    auto out = torch::empty_like(x);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Decide constant-memory path
    bool use_const = (C <= 4096);

    // Vectorized path if aligned and at least 4 elements
    bool vec_ok = (total_elems >= 4) &&
                  ((((uintptr_t)x.data_ptr<float>()) & 0xF) == 0) &&
                  ((((uintptr_t)out.data_ptr<float>()) & 0xF) == 0);

    // Grid sizing:
    // Keep enough CTAs to saturate memory (latency hiding), but avoid gigantic grids.
    int dev = at::cuda::current_device();
    int sm_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    const int threads = 256;

    // Empirical: for mem-bound elementwise, ~8-20 CTAs/SM is usually enough.
    int blocks_cap = sm_count * 16;
    if (blocks_cap < 1) blocks_cap = 1;
    if (blocks_cap > 65535) blocks_cap = 65535;

    // For tail scalar kernel, use fewer blocks (small leftover)
    int tail_blocks = sm_count * 4;
    if (tail_blocks < 1) tail_blocks = 1;
    if (tail_blocks > 65535) tail_blocks = 65535;

    // Prepare fast divmod descriptor (host-side) and pass by value
    FastDivmodU32 fdm = FastDivmodU32::make((uint32_t)C);

    // Optional: nontemporal stores only for very large outputs (reduce cache pollution)
    bool use_nt = (total_elems >= (1ll << 26)); // ~67M floats

    if (use_const) {
        // Copy add vector into constant memory (async on the same stream)
        cudaMemcpyToSymbolAsync(ADD_C_CONST, add_c.data_ptr<float>(), (size_t)C * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);

        if (vec_ok) {
            int64_t total4 = total_elems >> 2;
            int blocks = (int)((total4 + threads - 1) / threads);
            if (blocks > blocks_cap) blocks = blocks_cap;
            if (blocks < 1) blocks = 1;

            if (use_nt) {
                fused_flatten_vec4_kernel<true, true><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                    total_elems, C, spatial, fdm,
                    (float)neg_slope, (float)clamp_lo, (float)clamp_hi
                );
            } else {
                fused_flatten_vec4_kernel<true, false><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                    total_elems, C, spatial, fdm,
                    (float)neg_slope, (float)clamp_lo, (float)clamp_hi
                );
            }

            // Tail
            int64_t start = (total4 << 2);
            if (start < total_elems) {
                fused_tail_scalar_kernel<true><<<tail_blocks, threads, 0, stream>>>(
                    x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                    start, total_elems, C, spatial, fdm,
                    (float)neg_slope, (float)clamp_lo, (float)clamp_hi
                );
            }
        } else {
            // Scalar-only (includes full coverage)
            int blocks = (int)((total_elems + threads - 1) / threads);
            if (blocks > blocks_cap) blocks = blocks_cap;
            if (blocks < 1) blocks = 1;

            fused_tail_scalar_kernel<true><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                0, total_elems, C, spatial, fdm,
                (float)neg_slope, (float)clamp_lo, (float)clamp_hi
            );
        }
    } else {
        if (vec_ok) {
            int64_t total4 = total_elems >> 2;
            int blocks = (int)((total4 + threads - 1) / threads);
            if (blocks > blocks_cap) blocks = blocks_cap;
            if (blocks < 1) blocks = 1;

            if (use_nt) {
                fused_flatten_vec4_kernel<false, true><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                    total_elems, C, spatial, fdm,
                    (float)neg_slope, (float)clamp_lo, (float)clamp_hi
                );
            } else {
                fused_flatten_vec4_kernel<false, false><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                    total_elems, C, spatial, fdm,
                    (float)neg_slope, (float)clamp_lo, (float)clamp_hi
                );
            }

            int64_t start = ((total_elems >> 2) << 2);
            if (start < total_elems) {
                fused_tail_scalar_kernel<false><<<tail_blocks, threads, 0, stream>>>(
                    x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                    start, total_elems, C, spatial, fdm,
                    (float)neg_slope, (float)clamp_lo, (float)clamp_hi
                );
            }
        } else {
            int blocks = (int)((total_elems + threads - 1) / threads);
            if (blocks > blocks_cap) blocks = blocks_cap;
            if (blocks < 1) blocks = 1;

            fused_tail_scalar_kernel<false><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), add_c.data_ptr<float>(), out.data_ptr<float>(),
                0, total_elems, C, spatial, fdm,
                (float)neg_slope, (float)clamp_lo, (float)clamp_hi
            );
        }
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor fused_conv3d_postops_cuda(torch::Tensor x, torch::Tensor add_tensor, double neg_slope, double clamp_lo, double clamp_hi);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_leakyrelu_add_clamp_gelu_opt7_fastdiv_gridcap_tail",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_conv3d_postops_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv3d computed by PyTorch/cuDNN, then a fused CUDA kernel applies:
    LeakyReLU(0.2) -> Add(sum_tensor) -> Clamp[-1,1] -> GELU.
    Falls back to PyTorch ops if not CUDA/float32/contiguous or shape mismatch.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.custom_ops_lib = custom_ops_lib
        self.neg_slope = 0.2
        self.clamp_lo = -1.0
        self.clamp_hi = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or x.dtype != torch.float32:
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = x + self.sum_tensor
            x = torch.clamp(x, min=self.clamp_lo, max=self.clamp_hi)
            x = F.gelu(x)
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        add_t = self.sum_tensor
        if (add_t.dim() != 4) or (add_t.size(1) != 1) or (add_t.size(2) != 1) or (add_t.size(3) != 1):
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = x + self.sum_tensor
            x = torch.clamp(x, min=self.clamp_lo, max=self.clamp_hi)
            x = F.gelu(x)
            return x

        if add_t.size(0) != x.size(1):
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = x + self.sum_tensor
            x = torch.clamp(x, min=self.clamp_lo, max=self.clamp_hi)
            x = F.gelu(x)
            return x

        if not add_t.is_cuda:
            add_t = add_t.to(device=x.device)
        if add_t.dtype != torch.float32:
            add_t = add_t.float()
        if not add_t.is_contiguous():
            add_t = add_t.contiguous()

        return self.custom_ops_lib.fused_conv3d_postops_cuda(
            x, add_t, float(self.neg_slope), float(self.clamp_lo), float(self.clamp_hi)
        )


batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]