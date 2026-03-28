import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: ConvTranspose3d forward + Softmax(dim=1) + Sigmoid
# Improvements over baseline:
#  - Specialized fast convT kernel for (Cin=32,Cout=64,K=3,stride=2,pad=1,outpad=1) with W-tiling (2 outputs/thread)
#    and __launch_bounds__ to reduce registers / increase occupancy.
#  - Specialized softmax+sigmoid for Cout=64 using warp-shuffle reductions (2 warps compute 2 spatial positions per CTA),
#    eliminating per-thread shared-memory reductions and most __syncthreads().
#  - Generic fallbacks preserved for all other shapes.
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
#include <math_constants.h>

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// -------------------- Generic conv-transpose forward (fallback) --------------------
__global__ void conv_transpose3d_forward_bias_kernel_generic(
    const float* __restrict__ x,      // [N, Cin, Din, Hin, Win]
    const float* __restrict__ w,      // [Cin, Cout, Kd, Kh, Kw]
    const float* __restrict__ b,      // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, Dout, Hout, Wout]
    int N, int Cin, int Cout,
    int Din, int Hin, int Win,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int Dout, int Hout, int Wout,
    int has_bias
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)Dout * (int64_t)Hout * (int64_t)Wout;
    if (idx >= total) return;

    int64_t t = idx;
    int ow = (int)(t % Wout); t /= Wout;
    int oh = (int)(t % Hout); t /= Hout;
    int od = (int)(t % Dout); t /= Dout;
    int co = (int)(t % Cout); t /= Cout;
    int n  = (int)t;

    float acc = has_bias ? ldg_f(b + co) : 0.0f;

    const int64_t x_ci_stride = (int64_t)Din * (int64_t)Hin * (int64_t)Win;
    const int64_t w_ci_stride = (int64_t)Cout * (int64_t)Kd * (int64_t)Kh * (int64_t)Kw;
    const int64_t w_co_stride = (int64_t)Kd * (int64_t)Kh * (int64_t)Kw;

    for (int kd = 0; kd < Kd; ++kd) {
        int a = od + pad_d - kd;
        if (a < 0) continue;
        if (a % stride_d != 0) continue;
        int id = a / stride_d;
        if ((unsigned)id >= (unsigned)Din) continue;

        for (int kh = 0; kh < Kh; ++kh) {
            int bb = oh + pad_h - kh;
            if (bb < 0) continue;
            if (bb % stride_h != 0) continue;
            int ih = bb / stride_h;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            for (int kw = 0; kw < Kw; ++kw) {
                int cc = ow + pad_w - kw;
                if (cc < 0) continue;
                if (cc % stride_w != 0) continue;
                int iw = cc / stride_w;
                if ((unsigned)iw >= (unsigned)Win) continue;

                int64_t x_spatial = ((int64_t)id * (int64_t)Hin + (int64_t)ih) * (int64_t)Win + (int64_t)iw;
                int64_t x_n_base = ((int64_t)n * (int64_t)Cin) * x_ci_stride + x_spatial;

                int64_t k_off = ((int64_t)kd * (int64_t)Kh + (int64_t)kh) * (int64_t)Kw + (int64_t)kw;
                int64_t w_co_base = (int64_t)co * w_co_stride + k_off;

                for (int ci = 0; ci < Cin; ++ci) {
                    float xv = ldg_f(x + x_n_base + (int64_t)ci * x_ci_stride);
                    float wv = ldg_f(w + (int64_t)ci * w_ci_stride + w_co_base);
                    acc = fmaf(xv, wv, acc);
                }
            }
        }
    }
    y[idx] = acc;
}

// -------------------- Specialized conv-transpose for common config --------------------
// Assumes: Cin=32, Cout=64, K=3, stride=2, pad=1, outpad=1, contiguous NCDHW.
// Parallelization: 1D grid over (n, co, od, oh, ow_pair) where each thread computes 2 W outputs.
__global__ __launch_bounds__(128, 3) void convT3d_s2p1_k3_c32_c64_w2_kernel(
    const float* __restrict__ x,   // [N,32,Din,Hin,Win]
    const float* __restrict__ w,   // [32,64,3,3,3]
    const float* __restrict__ b,   // [64] or nullptr
    float* __restrict__ y,         // [N,64,Dout,Hout,Wout]
    int N, int Din, int Hin, int Win,
    int Dout, int Hout, int Wout,
    int has_bias
) {
    // Flattened index over elements where each element corresponds to a pair of ow (ow, ow+1)
    // total_pairs = N * 64 * Dout * Hout * ceil(Wout/2)
    int tid = (int)threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)tid;

    int Wpairs = (Wout + 1) >> 1;
    int64_t total_pairs = (int64_t)N * 64LL * (int64_t)Dout * (int64_t)Hout * (int64_t)Wpairs;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    int ow2 = (int)(t % Wpairs); t /= Wpairs;
    int oh  = (int)(t % Hout);   t /= Hout;
    int od  = (int)(t % Dout);   t /= Dout;
    int co  = (int)(t % 64);     t /= 64;
    int n   = (int)t;

    int ow0 = ow2 << 1;
    int ow1 = ow0 + 1;

    float acc0 = has_bias ? ldg_f(b + co) : 0.0f;
    float acc1 = acc0;

    // Strides (all fit in 32-bit for typical sizes, but use int64 for pointer arithmetic)
    const int64_t x_ci_stride = (int64_t)Din * (int64_t)Hin * (int64_t)Win;
    const int64_t x_n_base0 = ((int64_t)n * 32LL) * x_ci_stride;

    // Weight layout: [ci, co, kd, kh, kw]
    const int64_t w_ci_stride = 64LL * 27LL; // Cout * K3^3
    const int64_t w_co_base = (int64_t)co * 27LL;

    // Iterate kernel taps (kd,kh,kw): output->input mapping for stride=2, pad=1:
    // a = od + 1 - kd, require even; id = a/2 in [0, Din)
    // similarly for h,w.
#pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        int a = od + 1 - kd;
        if ((unsigned)a > 0x7fffffffU) continue; // negative
        if (a & 1) continue;
        int id = a >> 1;
        if ((unsigned)id >= (unsigned)Din) continue;

#pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int bb = oh + 1 - kh;
            if ((unsigned)bb > 0x7fffffffU) continue;
            if (bb & 1) continue;
            int ih = bb >> 1;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            // Precompute x spatial base (without iw) for both ow0/ow1 paths
            int64_t x_dh_base = ((int64_t)id * (int64_t)Hin + (int64_t)ih) * (int64_t)Win;

#pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                // ow0
                int cc0 = ow0 + 1 - kw;
                bool ok0 = ((unsigned)cc0 <= 0x7fffffffU) && ((cc0 & 1) == 0);
                int iw0 = cc0 >> 1;

                // ow1
                int cc1 = ow1 + 1 - kw;
                bool ok1 = ((unsigned)cc1 <= 0x7fffffffU) && ((cc1 & 1) == 0);
                int iw1 = cc1 >> 1;

                // Skip if both invalid
                if ((!ok0 || (unsigned)iw0 >= (unsigned)Win) && (!ok1 || (unsigned)iw1 >= (unsigned)Win)) continue;

                int64_t k_off = ((int64_t)kd * 9LL + (int64_t)kh * 3LL + (int64_t)kw);
                int64_t w_off_base = w_co_base + k_off;

                // Accumulate for valid iw0/iw1
                if (ok0 && (unsigned)iw0 < (unsigned)Win) {
                    int64_t x_spatial0 = x_dh_base + (int64_t)iw0;
                    const float* __restrict__ x_ptr0 = x + x_n_base0 + x_spatial0;
                    // ci loop: 32
#pragma unroll
                    for (int ci = 0; ci < 32; ++ci) {
                        float xv = ldg_f(x_ptr0 + (int64_t)ci * x_ci_stride);
                        float wv = ldg_f(w + (int64_t)ci * w_ci_stride + w_off_base);
                        acc0 = fmaf(xv, wv, acc0);
                    }
                }
                if (ok1 && (unsigned)iw1 < (unsigned)Win && ow1 < Wout) {
                    int64_t x_spatial1 = x_dh_base + (int64_t)iw1;
                    const float* __restrict__ x_ptr1 = x + x_n_base0 + x_spatial1;
#pragma unroll
                    for (int ci = 0; ci < 32; ++ci) {
                        float xv = ldg_f(x_ptr1 + (int64_t)ci * x_ci_stride);
                        float wv = ldg_f(w + (int64_t)ci * w_ci_stride + w_off_base);
                        acc1 = fmaf(xv, wv, acc1);
                    }
                }
            }
        }
    }

    // Store outputs (flattened contiguous N,C,D,H,W)
    int64_t out_spatial = ((int64_t)od * (int64_t)Hout + (int64_t)oh) * (int64_t)Wout;
    int64_t out_base = ((((int64_t)n * 64LL + (int64_t)co) * (int64_t)Dout) * (int64_t)Hout) * (int64_t)Wout
                       + out_spatial;

    if (ow0 < Wout) y[out_base + (int64_t)ow0] = acc0;
    if (ow1 < Wout) y[out_base + (int64_t)ow1] = acc1;
}

// -------------------- Softmax+sigmoid --------------------

// Generic (fallback) softmax(dim=1) + sigmoid using shared memory block reduction
__global__ void softmax_sigmoid_dim1_kernel_flat_generic(
    const float* __restrict__ inp,  // [N, C, D, H, W] contiguous
    float* __restrict__ out,        // [N, C, D, H, W] contiguous
    int N, int C, int D, int H, int W
) {
    int64_t p = (int64_t)blockIdx.x;
    int64_t P = (int64_t)N * (int64_t)D * (int64_t)H * (int64_t)W;
    if (p >= P) return;

    int tid = threadIdx.x;

    int64_t t = p;
    int w_ = (int)(t % W); t /= W;
    int h_ = (int)(t % H); t /= H;
    int d_ = (int)(t % D); t /= D;
    int n_ = (int)t;

    const int64_t c_stride = (int64_t)D * (int64_t)H * (int64_t)W;
    const int64_t base = ((((int64_t)n_ * (int64_t)C) * (int64_t)D + (int64_t)d_) * (int64_t)H + (int64_t)h_) * (int64_t)W + (int64_t)w_;

    float local_max = -CUDART_INF_F;
    for (int c = tid; c < C; c += blockDim.x) {
        float v = ldg_f(inp + base + (int64_t)c * c_stride);
        local_max = fmaxf(local_max, v);
    }

    extern __shared__ float smem[];
    smem[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        __syncthreads();
    }
    float maxv = smem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float v = ldg_f(inp + base + (int64_t)c * c_stride);
        local_sum += __expf(v - maxv);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }
    float sumv = smem[0];
    __syncthreads();

    for (int c = tid; c < C; c += blockDim.x) {
        float v = ldg_f(inp + base + (int64_t)c * c_stride);
        float sm = __expf(v - maxv) / sumv;
        out[base + (int64_t)c * c_stride] = sigmoidf_fast(sm);
    }
}

// Warp-reduction helpers
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// Specialized softmax+sigmoid for C=64: one warp handles one spatial position.
// Block has 2 warps => 2 positions per block for better occupancy.
__global__ __launch_bounds__(64, 4) void softmax_sigmoid_dim1_c64_warp2_kernel(
    const float* __restrict__ inp,  // [N,64,D,H,W]
    float* __restrict__ out,        // [N,64,D,H,W]
    int64_t P,                      // N*D*H*W
    int D, int H, int W
) {
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;            // 0..1
    int lane = tid & 31;            // 0..31

    int64_t p = (int64_t)blockIdx.x * 2LL + (int64_t)warp;
    if (p >= P) return;

    // decode p -> (n,d,h,w)
    int64_t t = p;
    int w_ = (int)(t % W); t /= W;
    int h_ = (int)(t % H); t /= H;
    int d_ = (int)(t % D); t /= D;
    int n_ = (int)t;

    const int64_t c_stride = (int64_t)D * (int64_t)H * (int64_t)W;
    const int64_t base = ((((int64_t)n_ * 64LL) * (int64_t)D + (int64_t)d_) * (int64_t)H + (int64_t)h_) * (int64_t)W + (int64_t)w_;

    // each lane handles 2 channels: lane and lane+32
    float v0 = ldg_f(inp + base + (int64_t)lane * c_stride);
    float v1 = ldg_f(inp + base + (int64_t)(lane + 32) * c_stride);

    float local_max = fmaxf(v0, v1);
    float maxv = warp_reduce_max(local_max);
    maxv = __shfl_sync(0xffffffff, maxv, 0);

    float e0 = __expf(v0 - maxv);
    float e1 = __expf(v1 - maxv);
    float local_sum = e0 + e1;
    float sumv = warp_reduce_sum(local_sum);
    sumv = __shfl_sync(0xffffffff, sumv, 0);
    float inv_sum = 1.0f / sumv;

    float sm0 = e0 * inv_sum;
    float sm1 = e1 * inv_sum;

    out[base + (int64_t)lane * c_stride] = sigmoidf_fast(sm0);
    out[base + (int64_t)(lane + 32) * c_stride] = sigmoidf_fast(sm1);
}

// -------------------- C++ interface --------------------

torch::Tensor conv_transpose3d_softmax_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t outpad_d, int64_t outpad_h, int64_t outpad_w
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be 5D [Cin,Cout,Kd,Kh,Kw]");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    torch::Tensor b;
    int has_bias = 0;
    const float* b_ptr = nullptr;
    if (b_opt.has_value()) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA if provided");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1, "bias must be 1D [Cout]");
        b = b.contiguous();
        has_bias = 1;
        b_ptr = b.data_ptr<float>();
    }

    const int64_t N   = x_c.size(0);
    const int64_t Cin = x_c.size(1);
    const int64_t Din = x_c.size(2);
    const int64_t Hin = x_c.size(3);
    const int64_t Win = x_c.size(4);

    const int64_t wCin = w_c.size(0);
    const int64_t Cout = w_c.size(1);
    const int64_t Kd = w_c.size(2);
    const int64_t Kh = w_c.size(3);
    const int64_t Kw = w_c.size(4);

    TORCH_CHECK(wCin == Cin, "Weight Cin must match input channels");
    if (has_bias) TORCH_CHECK(b.numel() == Cout, "bias size must equal Cout");

    TORCH_CHECK(stride_d >= 1 && stride_h >= 1 && stride_w >= 1, "stride must be >= 1");
    TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0, "padding must be >= 0");
    TORCH_CHECK(outpad_d >= 0 && outpad_d < stride_d, "output_padding_d must be in [0, stride_d-1]");
    TORCH_CHECK(outpad_h >= 0 && outpad_h < stride_h, "output_padding_h must be in [0, stride_h-1]");
    TORCH_CHECK(outpad_w >= 0 && outpad_w < stride_w, "output_padding_w must be in [0, stride_w-1]");

    const int64_t Dout = (Din - 1) * stride_d - 2 * pad_d + Kd + outpad_d;
    const int64_t Hout = (Hin - 1) * stride_h - 2 * pad_h + Kh + outpad_h;
    const int64_t Wout = (Win - 1) * stride_w - 2 * pad_w + Kw + outpad_w;
    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "Computed output shape must be positive");

    auto tmp = torch::empty({N, Cout, Dout, Hout, Wout}, x_c.options());
    auto out = torch::empty({N, Cout, Dout, Hout, Wout}, x_c.options());

    // Fast convT path only for the prompt/common configuration.
    bool fast_convt = (Cin == 32 && Cout == 64 &&
                       Kd == 3 && Kh == 3 && Kw == 3 &&
                       stride_d == 2 && stride_h == 2 && stride_w == 2 &&
                       pad_d == 1 && pad_h == 1 && pad_w == 1 &&
                       outpad_d == 1 && outpad_h == 1 && outpad_w == 1);

    if (fast_convt) {
        int Wpairs = (int)((Wout + 1) >> 1);
        int64_t total_pairs = N * 64LL * Dout * Hout * (int64_t)Wpairs;
        const int threads = 128;
        int blocks = (int)((total_pairs + threads - 1) / threads);

        convT3d_s2p1_k3_c32_c64_w2_kernel<<<blocks, threads>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            tmp.data_ptr<float>(),
            (int)N, (int)Din, (int)Hin, (int)Win,
            (int)Dout, (int)Hout, (int)Wout,
            has_bias
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        int64_t total = N * Cout * Dout * Hout * Wout;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);

        conv_transpose3d_forward_bias_kernel_generic<<<blocks, threads>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            tmp.data_ptr<float>(),
            (int)N, (int)Cin, (int)Cout,
            (int)Din, (int)Hin, (int)Win,
            (int)Kd, (int)Kh, (int)Kw,
            (int)stride_d, (int)stride_h, (int)stride_w,
            (int)pad_d, (int)pad_h, (int)pad_w,
            (int)Dout, (int)Hout, (int)Wout,
            has_bias
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Softmax+sigmoid: specialized for Cout=64, else fallback.
    if (Cout == 64) {
        int64_t P = N * Dout * Hout * Wout;
        // 2 warps per block => 64 threads
        const int threads = 64;
        int blocks = (int)((P + 2 - 1) / 2); // 2 positions per block
        softmax_sigmoid_dim1_c64_warp2_kernel<<<blocks, threads>>>(
            tmp.data_ptr<float>(),
            out.data_ptr<float>(),
            P,
            (int)Dout, (int)Hout, (int)Wout
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        int64_t P = N * Dout * Hout * Wout;
        int threads = 128;
        if (Cout <= 64) threads = 64;
        else if (Cout <= 128) threads = 128;
        else threads = 256;
        if (threads > 256) threads = 256;

        dim3 grid((unsigned)P, 1, 1);
        size_t shmem = (size_t)threads * sizeof(float);

        softmax_sigmoid_dim1_kernel_flat_generic<<<grid, threads, shmem>>>(
            tmp.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N, (int)Cout, (int)Dout, (int)Hout, (int)Wout
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose3d_softmax_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t outpad_d, int64_t outpad_h, int64_t outpad_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_softmax_sigmoid_v4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_softmax_sigmoid_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement for:
      ConvTranspose3d -> Softmax(dim=1) -> Sigmoid
    using optimized custom CUDA kernels.

    Supports:
      - groups=1 only
      - float32 CUDA inputs
      - bias optional
      - stride/padding/output_padding as int or 3-tuple
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        def _triple(v):
            if isinstance(v, (tuple, list)):
                if len(v) != 3:
                    raise ValueError("Expected 3-tuple for 3D params")
                return (int(v[0]), int(v[1]), int(v[2]))
            v = int(v)
            return (v, v, v)

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)

        # Weight layout matches PyTorch ConvTranspose3d: [Cin, Cout, Kd, Kh, Kw]
        w = torch.empty(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.empty(self.out_channels, dtype=torch.float32)
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("ModelNew expects CUDA input")
        if x.dtype != torch.float32:
            raise ValueError("ModelNew expects float32 input")

        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        opd, oph, opw = self.output_padding

        b_opt = self.bias if self.bias is not None else None
        return self.custom_ops.conv_transpose3d_softmax_sigmoid_cuda(
            x, self.weight, b_opt,
            sd, sh, sw,
            pd, ph, pw,
            opd, oph, opw
        )