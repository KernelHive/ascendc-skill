import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension: fused SRM forward (eval-mode BN semantics)
# ----------------------------

srm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  #define LDG(x) __ldg(x)
#else
  #define LDG(x) (*(x))
#endif

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ void warp_welford_combine(float &mean, float &m2, float &count) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mean_b  = __shfl_down_sync(mask, mean,  offset);
        float m2_b    = __shfl_down_sync(mask, m2,    offset);
        float count_b = __shfl_down_sync(mask, count, offset);

        if (count_b > 0.0f) {
            float delta = mean_b - mean;
            float tot = count + count_b;
            mean = mean + delta * (count_b / tot);
            m2   = m2 + m2_b + delta * delta * (count * count_b / tot);
            count = tot;
        }
    }
}

static __device__ __forceinline__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

static __device__ __forceinline__ void ld_global_v4(float &a, float &b, float &c, float &d, const float* ptr) {
    // ptr must be 16B aligned
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                 : "l"(ptr));
}

static __device__ __forceinline__ void st_global_v4(float* ptr, float a, float b, float c, float d) {
    // ptr must be 16B aligned
    asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "f"(a), "f"(b), "f"(c), "f"(d));
}

// HW=49 fast path: WARPS_PER_BLOCK warps per block. Persistent grid-stride over channel tiles.
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void srm_forward_hw49_warps_persistent_kernel(
    const float* __restrict__ x,      // [B,C,HW] flattened
    float* __restrict__ out,          // [B,C,HW] flattened
    const float* __restrict__ w2,     // [C,2]
    const float* __restrict__ bn_w,   // [C]
    const float* __restrict__ bn_b,   // [C]
    const float* __restrict__ rm,     // [C]
    const float* __restrict__ rv,     // [C]
    int B, int C,
    float eps)
{
    constexpr int HW = 49;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int b = (int)blockIdx.x;
    int tile0 = (int)blockIdx.y;

    int grid_y = (int)gridDim.y;

    for (int tile = tile0; tile < (C + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK; tile += grid_y) {
        int c = tile * WARPS_PER_BLOCK + warp;
        if (warp >= WARPS_PER_BLOCK || c >= C) continue;

        int base = (b * C + c) * HW;

        // Welford over 49: each lane loads lane and lane+32 if valid.
        float mean = 0.0f, m2 = 0.0f, count = 0.0f;

        int i0 = lane;
        float v0 = 0.0f;
        if (i0 < HW) {
            v0 = x[base + i0];
            mean = v0;
            m2 = 0.0f;
            count = 1.0f;
        }
        int i1 = lane + 32;
        float v1 = 0.0f;
        if (i1 < HW) {
            v1 = x[base + i1];
            if (count == 0.0f) {
                mean = v1;
                m2 = 0.0f;
                count = 1.0f;
            } else {
                float delta = v1 - mean;
                float new_count = count + 1.0f;
                mean += delta / new_count;
                float delta2 = v1 - mean;
                m2 += delta * delta2;
                count = new_count;
            }
        }

        warp_welford_combine(mean, m2, count);
        float mean_all = __shfl_sync(0xffffffffu, mean, 0);
        float m2_all   = __shfl_sync(0xffffffffu, m2,   0);

        float var_unbiased = m2_all * (1.0f / 48.0f);
        if (var_unbiased < 0.0f) var_unbiased = 0.0f;
        float std_unbiased = sqrtf(var_unbiased);

        float w0 = LDG(&w2[c * 2 + 0]);
        float w1 = LDG(&w2[c * 2 + 1]);
        float z = fmaf(w0, mean_all, w1 * std_unbiased);

        float rm_c = LDG(&rm[c]);
        float rv_c = LDG(&rv[c]);
        float inv_std = rsqrtf(rv_c + eps);
        float zhat = (z - rm_c) * inv_std;

        float bnw = LDG(&bn_w[c]);
        float bnb = LDG(&bn_b[c]);
        float zbn = fmaf(zhat, bnw, bnb);

        float g = sigmoid_fast(zbn);
        g = __shfl_sync(0xffffffffu, g, 0);

        const float* xptr = x + base;
        float* optr = out + base;

        bool aligned = is_aligned_16(xptr) && is_aligned_16(optr);

        if (aligned) {
            // 49 floats = 12*float4 (48) + tail.
            if (lane < 12) {
                const float* xp = xptr + lane * 4;
                float a, b, ccc, d;
                ld_global_v4(a, b, ccc, d, xp);
                a *= g; b *= g; ccc *= g; d *= g;
                float* op = optr + lane * 4;
                st_global_v4(op, a, b, ccc, d);
            }
            if (lane == 0) {
                optr[48] = xptr[48] * g;
            }
        } else {
            if (i0 < HW) optr[i0] = v0 * g;
            if (i1 < HW) optr[i1] = v1 * g;
        }
    }
}

// Generic fallback (for other HW): block reduction with minimal barriers.
template<int THREADS>
__global__ void srm_forward_generic_kernel(
    const float* __restrict__ x,      // [B,C,HW]
    float* __restrict__ out,          // [B,C,HW]
    const float* __restrict__ w2,     // [C,2]
    const float* __restrict__ bn_w,   // [C]
    const float* __restrict__ bn_b,   // [C]
    const float* __restrict__ rm,     // [C]
    const float* __restrict__ rv,     // [C]
    int C, int HW,
    float eps)
{
    int bc = (int)blockIdx.x;
    int b = bc / C;
    int c = bc - b * C;
    int base = (b * C + c) * HW;

    float mean = 0.0f, m2 = 0.0f, count = 0.0f;

    for (int i = (int)threadIdx.x; i < HW; i += THREADS) {
        float v = x[base + i];
        if (count == 0.0f) {
            mean = v;
            m2 = 0.0f;
            count = 1.0f;
        } else {
            float delta = v - mean;
            float new_count = count + 1.0f;
            mean += delta / new_count;
            float delta2 = v - mean;
            m2 += delta * delta2;
            count = new_count;
        }
    }

    __shared__ float s_mean[32];
    __shared__ float s_m2[32];
    __shared__ float s_count[32];

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    warp_welford_combine(mean, m2, count);

    if (lane == 0) {
        s_mean[warp] = mean;
        s_m2[warp] = m2;
        s_count[warp] = count;
    }
    __syncthreads();

    float mean_all = 0.0f, m2_all = 0.0f, count_all = 0.0f;

    if (warp == 0) {
        int nwarps = (THREADS + 31) / 32;
        if (lane < nwarps) {
            mean_all  = s_mean[lane];
            m2_all    = s_m2[lane];
            count_all = s_count[lane];
        }
        warp_welford_combine(mean_all, m2_all, count_all);
        mean_all = __shfl_sync(0xffffffffu, mean_all, 0);
        m2_all   = __shfl_sync(0xffffffffu, m2_all,   0);
        if (lane == 0) {
            s_mean[0] = mean_all;
            s_m2[0] = m2_all;
        }
    }
    __syncthreads();

    float final_mean = s_mean[0];
    float final_m2   = s_m2[0];

    float var_unbiased = 0.0f;
    if (HW > 1) {
        var_unbiased = final_m2 / (float)(HW - 1);
        if (var_unbiased < 0.0f) var_unbiased = 0.0f;
    }
    float std_unbiased = sqrtf(var_unbiased);

    float w0 = LDG(&w2[c * 2 + 0]);
    float w1 = LDG(&w2[c * 2 + 1]);
    float z = fmaf(w0, final_mean, w1 * std_unbiased);

    float rm_c = LDG(&rm[c]);
    float rv_c = LDG(&rv[c]);
    float inv_std = rsqrtf(rv_c + eps);
    float zhat = (z - rm_c) * inv_std;

    float bnw = LDG(&bn_w[c]);
    float bnb = LDG(&bn_b[c]);
    float zbn = fmaf(zhat, bnw, bnb);

    float g = sigmoid_fast(zbn);

    for (int i = (int)threadIdx.x; i < HW; i += THREADS) {
        out[base + i] = x[base + i] * g;
    }
}

torch::Tensor srm_forward_cuda(
    torch::Tensor x,                 // [B,C,H,W]
    torch::Tensor conv_w,            // [C,1,2]
    torch::Tensor bn_w,              // [C]
    torch::Tensor bn_b,              // [C]
    torch::Tensor running_mean,      // [C]
    torch::Tensor running_var,       // [C]
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B,C,H,W)");

    TORCH_CHECK(conv_w.is_cuda() && conv_w.dtype() == torch::kFloat32, "conv_w must be CUDA float32");
    TORCH_CHECK(bn_w.is_cuda() && bn_w.dtype() == torch::kFloat32, "bn_w must be CUDA float32");
    TORCH_CHECK(bn_b.is_cuda() && bn_b.dtype() == torch::kFloat32, "bn_b must be CUDA float32");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32, "running_mean must be CUDA float32");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32, "running_var must be CUDA float32");

    TORCH_CHECK(conv_w.is_contiguous(), "conv_w must be contiguous");
    TORCH_CHECK(bn_w.is_contiguous(), "bn_w must be contiguous");
    TORCH_CHECK(bn_b.is_contiguous(), "bn_b must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int HW = H * W;

    TORCH_CHECK(conv_w.dim() == 3 && conv_w.size(0) == C && conv_w.size(1) == 1 && conv_w.size(2) == 2,
                "conv_w must have shape [C,1,2]");

    auto out = torch::empty_like(x);

    const float* w2 = (const float*)conv_w.data_ptr<float>();
    const float* xptr = (const float*)x.data_ptr<float>();
    float* optr = (float*)out.data_ptr<float>();

    if (HW == 49) {
        constexpr int WARPS_PER_BLOCK = 8; // 256 threads
        // A modest grid.y encourages persistence without launch explosion.
        // Using at most 32 tiles in Y is usually enough to fill SMs and reduce overhead.
        int tiles = (C + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int grid_y = tiles;
        if (grid_y > 32) grid_y = 32;
        dim3 blocks((unsigned)B, (unsigned)grid_y, 1);
        dim3 threads(WARPS_PER_BLOCK * 32, 1, 1);

        srm_forward_hw49_warps_persistent_kernel<WARPS_PER_BLOCK><<<blocks, threads>>>(
            xptr, optr, w2,
            (const float*)bn_w.data_ptr<float>(),
            (const float*)bn_b.data_ptr<float>(),
            (const float*)running_mean.data_ptr<float>(),
            (const float*)running_var.data_ptr<float>(),
            B, C, (float)eps
        );
    } else {
        const int blocks = B * C;
        constexpr int threads = 128;
        srm_forward_generic_kernel<threads><<<blocks, threads>>>(
            xptr, optr, w2,
            (const float*)bn_w.data_ptr<float>(),
            (const float*)bn_b.data_ptr<float>(),
            (const float*)running_mean.data_ptr<float>(),
            (const float*)running_var.data_ptr<float>(),
            C, HW, (float)eps
        );
    }

    return out;
}
"""

srm_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor srm_forward_cuda(
    torch::Tensor x,
    torch::Tensor conv_w,
    torch::Tensor bn_w,
    torch::Tensor bn_b,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_srm_opt_eval_v6_persist",
    cpp_sources=srm_cpp_source,
    cuda_sources=srm_cuda_source,
    functions=["srm_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    SRM forward optimized with a fused custom CUDA kernel.

    Notes:
    - This fused op matches BatchNorm1d *eval* semantics (uses running stats).
    - For training=True, falls back to PyTorch reference path.
    """
    def __init__(self, channel: int):
        super().__init__()
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, groups=channel, bias=False)
        self.bn = nn.BatchNorm1d(channel)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA input tensor")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        if self.training:
            b, c, h, w = x.shape
            mean = x.reshape(b, c, -1).mean(-1).unsqueeze(-1)
            std = x.reshape(b, c, -1).std(-1, unbiased=True).unsqueeze(-1)
            u = torch.cat([mean, std], dim=-1)
            z = self.cfc(u)
            z = self.bn(z)
            g = torch.sigmoid(z).reshape(b, c, 1, 1)
            return x * g.expand_as(x)

        conv_w = self.cfc.weight
        if conv_w.dtype != torch.float32:
            conv_w = conv_w.float()
        if not conv_w.is_contiguous():
            conv_w = conv_w.contiguous()

        bn_w = self.bn.weight
        bn_b = self.bn.bias
        if bn_w.dtype != torch.float32:
            bn_w = bn_w.float()
        if bn_b.dtype != torch.float32:
            bn_b = bn_b.float()
        if not bn_w.is_contiguous():
            bn_w = bn_w.contiguous()
        if not bn_b.is_contiguous():
            bn_b = bn_b.contiguous()

        rm = self.bn.running_mean
        rv = self.bn.running_var
        if rm.dtype != torch.float32:
            rm = rm.float()
        if rv.dtype != torch.float32:
            rv = rv.float()
        if not rm.is_contiguous():
            rm = rm.contiguous()
        if not rv.is_contiguous():
            rv = rv.contiguous()

        return self.custom_ops.srm_forward_cuda(
            x, conv_w, bn_w, bn_b, rm, rv, float(self.bn.eps)
        )