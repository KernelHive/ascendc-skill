import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
#include <vector>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

__device__ __forceinline__ float relu_f(float v) { return v > 0.f ? v : 0.f; }

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#else
__device__ __forceinline__ void prefetch_l2(const void* ) {}
#endif

// One warp computes 4 outputs for one batch row.
// Block = 128 threads (4 warps). Each block covers 4 batch rows (via warps) and a tile of 4 outputs (via blockIdx.x).
// No shared memory; rely on __ldg + L2 reuse of X across warps/blocks.
template <bool APPLY_RELU, bool VEC4>
__global__ __launch_bounds__(128, 4)
void linear_bias_relu_warp4_f32_kernel(
    const float* __restrict__ X,   // [B, ldX]
    const float* __restrict__ W,   // [O, I] row-major
    const float* __restrict__ Bptr,// [O]
    float* __restrict__ Y,         // [B, ldY]
    int B, int I, int O,
    int ldX, int ldY
) {
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;           // 0..3

    const int b  = (int)blockIdx.y * 4 + warp;
    const int o0 = (int)blockIdx.x * 4;

    if (b >= B) return;

    const float* xrow = X + (int64_t)b * ldX;

    // Prefetch a couple of cache lines of X and first weight rows to L2 (best-effort).
    if (lane == 0) {
        prefetch_l2(xrow);
        if (o0 < O) prefetch_l2(W + (int64_t)o0 * I);
    }

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    const float* w0 = (o0 + 0 < O) ? (W + (int64_t)(o0 + 0) * I) : nullptr;
    const float* w1 = (o0 + 1 < O) ? (W + (int64_t)(o0 + 1) * I) : nullptr;
    const float* w2 = (o0 + 2 < O) ? (W + (int64_t)(o0 + 2) * I) : nullptr;
    const float* w3 = (o0 + 3 < O) ? (W + (int64_t)(o0 + 3) * I) : nullptr;

    if constexpr (VEC4) {
        // I is multiple of 4, and pointers are assumed aligned enough for float4.
        const int I4 = I >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xrow);
        const float4* __restrict__ w0_4 = w0 ? reinterpret_cast<const float4*>(w0) : nullptr;
        const float4* __restrict__ w1_4 = w1 ? reinterpret_cast<const float4*>(w1) : nullptr;
        const float4* __restrict__ w2_4 = w2 ? reinterpret_cast<const float4*>(w2) : nullptr;
        const float4* __restrict__ w3_4 = w3 ? reinterpret_cast<const float4*>(w3) : nullptr;

        for (int k4 = lane; k4 < I4; k4 += 32) {
            float4 xv = __ldg(x4 + k4);
            if (w0_4) {
                float4 wv = __ldg(w0_4 + k4);
                acc0 = fmaf(xv.x, wv.x, acc0);
                acc0 = fmaf(xv.y, wv.y, acc0);
                acc0 = fmaf(xv.z, wv.z, acc0);
                acc0 = fmaf(xv.w, wv.w, acc0);
            }
            if (w1_4) {
                float4 wv = __ldg(w1_4 + k4);
                acc1 = fmaf(xv.x, wv.x, acc1);
                acc1 = fmaf(xv.y, wv.y, acc1);
                acc1 = fmaf(xv.z, wv.z, acc1);
                acc1 = fmaf(xv.w, wv.w, acc1);
            }
            if (w2_4) {
                float4 wv = __ldg(w2_4 + k4);
                acc2 = fmaf(xv.x, wv.x, acc2);
                acc2 = fmaf(xv.y, wv.y, acc2);
                acc2 = fmaf(xv.z, wv.z, acc2);
                acc2 = fmaf(xv.w, wv.w, acc2);
            }
            if (w3_4) {
                float4 wv = __ldg(w3_4 + k4);
                acc3 = fmaf(xv.x, wv.x, acc3);
                acc3 = fmaf(xv.y, wv.y, acc3);
                acc3 = fmaf(xv.z, wv.z, acc3);
                acc3 = fmaf(xv.w, wv.w, acc3);
            }
        }
    } else {
        // Generic scalar path with small unroll to increase ILP.
        int k = lane;
        for (; k + 1 < I; k += 32 * 2) {
            float x0 = __ldg(xrow + k);
            float x1 = __ldg(xrow + k + 32);

            if (w0) { acc0 = fmaf(x0, __ldg(w0 + k), acc0); acc0 = fmaf(x1, __ldg(w0 + k + 32), acc0); }
            if (w1) { acc1 = fmaf(x0, __ldg(w1 + k), acc1); acc1 = fmaf(x1, __ldg(w1 + k + 32), acc1); }
            if (w2) { acc2 = fmaf(x0, __ldg(w2 + k), acc2); acc2 = fmaf(x1, __ldg(w2 + k + 32), acc2); }
            if (w3) { acc3 = fmaf(x0, __ldg(w3 + k), acc3); acc3 = fmaf(x1, __ldg(w3 + k + 32), acc3); }
        }
        // Tail (handles both odd I and last iteration when I not multiple of 64)
        for (int kk = k; kk < I; kk += 32) {
            float xv = __ldg(xrow + kk);
            if (w0) acc0 = fmaf(xv, __ldg(w0 + kk), acc0);
            if (w1) acc1 = fmaf(xv, __ldg(w1 + kk), acc1);
            if (w2) acc2 = fmaf(xv, __ldg(w2 + kk), acc2);
            if (w3) acc3 = fmaf(xv, __ldg(w3 + kk), acc3);
        }
    }

    acc0 = warp_reduce_sum(acc0);
    acc1 = warp_reduce_sum(acc1);
    acc2 = warp_reduce_sum(acc2);
    acc3 = warp_reduce_sum(acc3);

    if (lane == 0) {
        float* yrow = Y + (int64_t)b * ldY;
        if (o0 + 0 < O) {
            float v = acc0 + (Bptr ? __ldg(Bptr + (o0 + 0)) : 0.f);
            if constexpr (APPLY_RELU) v = relu_f(v);
            yrow[o0 + 0] = v;
        }
        if (o0 + 1 < O) {
            float v = acc1 + (Bptr ? __ldg(Bptr + (o0 + 1)) : 0.f);
            if constexpr (APPLY_RELU) v = relu_f(v);
            yrow[o0 + 1] = v;
        }
        if (o0 + 2 < O) {
            float v = acc2 + (Bptr ? __ldg(Bptr + (o0 + 2)) : 0.f);
            if constexpr (APPLY_RELU) v = relu_f(v);
            yrow[o0 + 2] = v;
        }
        if (o0 + 3 < O) {
            float v = acc3 + (Bptr ? __ldg(Bptr + (o0 + 3)) : 0.f);
            if constexpr (APPLY_RELU) v = relu_f(v);
            yrow[o0 + 3] = v;
        }
    }
}

static inline bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

torch::Tensor deep_narrow_mlp_f32_cuda(
    torch::Tensor x,                       // [B, I0]
    std::vector<torch::Tensor> weights,    // L tensors: [O_l, I_l]
    std::vector<torch::Tensor> biases      // L tensors: [O_l]
) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, I]");
    TORCH_CHECK((int64_t)weights.size() >= 1, "weights must have at least 1 layer");
    TORCH_CHECK(biases.size() == weights.size(), "biases must be same length as weights (bias=True expected)");

    const int64_t L = (int64_t)weights.size();
    const int B = (int)x.size(0);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    int maxO = 0;
    for (int64_t l = 0; l < L; ++l) {
        auto w = weights[l];
        CHECK_INPUT(w);
        TORCH_CHECK(w.scalar_type() == at::kFloat, "weights must be float32");
        TORCH_CHECK(w.dim() == 2, "each weight must be 2D [O, I]");
        maxO = (int)std::max<int64_t>(maxO, w.size(0));
    }

    auto buf0 = torch::empty({B, maxO}, x.options());
    auto buf1 = torch::empty({B, maxO}, x.options());

    const float* cur_ptr = (const float*)x.data_ptr<float>();
    int curI = (int)x.size(1);
    int ldX = curI;  // tight for input x
    int ldY = maxO;  // stride for ping-pong buffers

    torch::Tensor out_full;

    dim3 block(128, 1, 1); // 4 warps
    for (int64_t l = 0; l < L; ++l) {
        auto w = weights[l];
        auto b = biases[l];
        CHECK_INPUT(b);
        TORCH_CHECK(b.scalar_type() == at::kFloat && b.dim() == 1, "bias must be float32 [O]");

        const int O = (int)w.size(0);
        const int I = (int)w.size(1);
        TORCH_CHECK(curI == I, "layer input feature mismatch");

        out_full = (l % 2 == 0) ? buf0 : buf1;

        dim3 grid((O + 4 - 1) / 4, (B + 4 - 1) / 4, 1);

        const bool apply_relu = (l != (L - 1));
        const float* Wptr = (const float*)w.data_ptr<float>();
        const float* Bptr = (const float*)b.data_ptr<float>();
        float* Yptr = (float*)out_full.data_ptr<float>();

        bool vec4_ok = ((I & 3) == 0) && is_aligned_16(cur_ptr) && is_aligned_16(Wptr);

        if (apply_relu) {
            if (vec4_ok) {
                linear_bias_relu_warp4_f32_kernel<true, true><<<grid, block, 0, stream>>>(
                    cur_ptr, Wptr, Bptr, Yptr, B, I, O, ldX, ldY
                );
            } else {
                linear_bias_relu_warp4_f32_kernel<true, false><<<grid, block, 0, stream>>>(
                    cur_ptr, Wptr, Bptr, Yptr, B, I, O, ldX, ldY
                );
            }
        } else {
            if (vec4_ok) {
                linear_bias_relu_warp4_f32_kernel<false, true><<<grid, block, 0, stream>>>(
                    cur_ptr, Wptr, Bptr, Yptr, B, I, O, ldX, ldY
                );
            } else {
                linear_bias_relu_warp4_f32_kernel<false, false><<<grid, block, 0, stream>>>(
                    cur_ptr, Wptr, Bptr, Yptr, B, I, O, ldX, ldY
                );
            }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        cur_ptr = (const float*)out_full.data_ptr<float>();
        curI = O;
        ldX = ldY; // ping-pong buffer stride
    }

    int O_last = (int)weights.back().size(0);
    return out_full.narrow(1, 0, O_last).contiguous();
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor deep_narrow_mlp_f32_cuda(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases
);
"""

_ext_name = "custom_ops_lib_deep_narrow_mlp_v6_warp4_128cta"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["deep_narrow_mlp_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Custom-op forward for deep_narrow_mlp:
      - CUDA float32 fast path using custom kernels
      - Fallback to nn.Sequential otherwise
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        cur = int(input_size)
        for h in hidden_layer_sizes:
            h = int(h)
            layers.append(nn.Linear(cur, h, bias=True))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, int(output_size), bias=True))
        self.network = nn.Sequential(*layers)

        self._linears = nn.ModuleList([m for m in self.network if isinstance(m, nn.Linear)])
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        try_custom = x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and x.dim() == 2
        if try_custom:
            weights = []
            biases = []
            ok = True
            for lin in self._linears:
                w = lin.weight
                b = lin.bias
                if w is None or b is None:
                    ok = False
                    break
                if not (w.is_cuda and w.dtype == torch.float32 and w.is_contiguous() and w.dim() == 2):
                    ok = False
                    break
                if not (b.is_cuda and b.dtype == torch.float32 and b.is_contiguous() and b.dim() == 1):
                    ok = False
                    break
                weights.append(w)
                biases.append(b)
            if ok:
                return self.custom_ops_lib.deep_narrow_mlp_f32_cuda(x, weights, biases)

        return self.network(x)