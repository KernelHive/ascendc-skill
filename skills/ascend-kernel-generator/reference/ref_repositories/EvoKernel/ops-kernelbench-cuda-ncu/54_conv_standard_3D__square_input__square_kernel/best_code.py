import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv_standard3d_square_input_square_kernel (v8) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline int64_t div_up_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ int64_t y_index_ncdhw(
    int n, int co, int od, int oh, int ow,
    int Cout, int Dout, int Hout, int Wout
) {
    return (((((int64_t)n * (int64_t)Cout + (int64_t)co) * (int64_t)Dout + (int64_t)od) * (int64_t)Hout + (int64_t)oh) * (int64_t)Wout + (int64_t)ow);
}

// -------------------------------------------
// Generic kernel (any Cin/K/stride/pad) FP32
// One thread -> one output element.
// -------------------------------------------
__global__ __launch_bounds__(256, 2)
void conv3d_fwd_ncdhw_f32_generic_kernel(
    const float* __restrict__ x,   // [N, Cin, D, H, W]
    const float* __restrict__ w,   // [Cout, Cin, K, K, K]
    float* __restrict__ y,         // [N, Cout, Dout, Hout, Wout]
    int N, int Cin, int D, int H, int W,
    int Cout, int K,
    int stride, int pad,
    int Dout, int Hout, int Wout
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Dout * Hout * Wout;
    if (idx >= total) return;

    int t = idx;
    int ow = t % Wout; t /= Wout;
    int oh = t % Hout; t /= Hout;
    int od = t % Dout; t /= Dout;
    int co = t % Cout; t /= Cout;
    int n  = t;

    int in_d0 = od * stride - pad;
    int in_h0 = oh * stride - pad;
    int in_w0 = ow * stride - pad;

    const int HW  = H * W;
    const int DHW = D * HW;
    const int KKK = K * K * K;

    float acc = 0.0f;

    int x_n_base = (n * Cin) * DHW;

    for (int ci = 0; ci < Cin; ++ci) {
        int x_c_base = x_n_base + ci * DHW;
        int w_base = (co * Cin + ci) * KKK;

        for (int kd = 0; kd < K; ++kd) {
            int id = in_d0 + kd;
            if ((unsigned)id >= (unsigned)D) continue;

            int x_d_base = x_c_base + id * HW;
            int w_kd_base = w_base + kd * (K * K);

            for (int kh = 0; kh < K; ++kh) {
                int ih = in_h0 + kh;
                if ((unsigned)ih >= (unsigned)H) continue;

                int x_h_base = x_d_base + ih * W;
                int w_kh_base = w_kd_base + kh * K;

#pragma unroll 1
                for (int kw = 0; kw < K; ++kw) {
                    int iw = in_w0 + kw;
                    if ((unsigned)iw >= (unsigned)W) continue;
                    float xv = ldg_f32(x + x_h_base + iw);
                    float wv = ldg_f32(w + w_kh_base + kw);
                    acc = fmaf(xv, wv, acc);
                }
            }
        }
    }

    y[y_index_ncdhw(n, co, od, oh, ow, Cout, Dout, Hout, Wout)] = acc;
}

// ---------------------------------------------------------
// Specialized fast path: Cin=3, K=3, stride=1, pad=0
// Block computes one OC tile of 8 (OCx8) over a grid-stride set of voxels.
// - 64 threads/block (2 warps) for higher occupancy.
// - First warp stages weights (up to 8*81 floats) into shared memory with full participation.
// - Each thread computes 1 voxel and accumulates 8 outputs.
// Notes:
// - Uses scalar stores to avoid alignment pitfalls.
// ---------------------------------------------------------
__global__ __launch_bounds__(64, 4)
void conv3d_fwd_k3_cin3_s1p0_oc8_wsmem_kernel(
    const float* __restrict__ x, // [N,3,D,H,W]
    const float* __restrict__ w, // [Cout,3,3,3,3]
    float* __restrict__ y,       // [N,Cout,D-2,H-2,W-2]
    int N, int D, int H, int W,
    int Cout
) {
    const int Dout = D - 2;
    const int Hout = H - 2;
    const int Wout = W - 2;

    const int oc0 = (int)(blockIdx.y * 8);

    // Shared weights: 8 output channels * 81 weights (Cin=3,K=3 => 81)
    __shared__ float sw[8 * 81];

    // cooperative weight load: first warp (32 threads) loads all needed weights
    // ensure all 32 threads participate with stride=32 (avoids underutilized load loops)
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        // number of valid channels in this tile
        int valid = Cout - oc0;
        if (valid > 8) valid = 8;
        int total = valid * 81;
        for (int i = lane; i < total; i += 32) {
            int oc = i / 81;
            int k  = i - oc * 81;
            sw[oc * 81 + k] = ldg_f32(w + (int64_t)(oc0 + oc) * 81 + k);
        }
    }
    __syncthreads();

    const int HW  = H * W;
    const int DHW = D * HW;
    const int out_vox = Dout * Hout * Wout;

    int64_t voxels = (int64_t)N * (int64_t)out_vox;
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x * (int64_t)blockDim.x;

    // valid oc lanes for tail
    int valid = Cout - oc0;
    if (valid > 8) valid = 8;
    if (valid <= 0) return;

    for (int64_t v = tid; v < voxels; v += step) {
        int t = (int)(v % out_vox);
        int n = (int)(v / out_vox);

        int ow = t % Wout; t /= Wout;
        int oh = t % Hout; t /= Hout;
        int od = t;

        // stride=1,pad=0
        const int in_d0 = od;
        const int in_h0 = oh;
        const int in_w0 = ow;

        const int64_t x_n_base = (int64_t)n * (int64_t)3 * (int64_t)DHW;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

#pragma unroll
        for (int ci = 0; ci < 3; ++ci) {
            const int64_t x_c_base = x_n_base + (int64_t)ci * (int64_t)DHW;
            const int w_ci_base = ci * 27;

#pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                const int id = in_d0 + kd;
                const int64_t base_d = x_c_base + (int64_t)id * (int64_t)HW;
                const int w_kd_base = w_ci_base + kd * 9;

#pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    const int ih = in_h0 + kh;
                    const int64_t base_h = base_d + (int64_t)ih * (int64_t)W + (int64_t)in_w0;
                    const int w_kh_base = w_kd_base + kh * 3;

                    // stream 3 values (kw=0..2)
                    float x0 = ldg_f32(x + base_h + 0);
                    float x1 = ldg_f32(x + base_h + 1);
                    float x2 = ldg_f32(x + base_h + 2);

                    // accumulate for valid lanes
                    if (valid > 0) {
                        const float* ws = sw + 0 * 81 + w_kh_base;
                        acc0 = fmaf(x0, ws[0], acc0); acc0 = fmaf(x1, ws[1], acc0); acc0 = fmaf(x2, ws[2], acc0);
                    }
                    if (valid > 1) {
                        const float* ws = sw + 1 * 81 + w_kh_base;
                        acc1 = fmaf(x0, ws[0], acc1); acc1 = fmaf(x1, ws[1], acc1); acc1 = fmaf(x2, ws[2], acc1);
                    }
                    if (valid > 2) {
                        const float* ws = sw + 2 * 81 + w_kh_base;
                        acc2 = fmaf(x0, ws[0], acc2); acc2 = fmaf(x1, ws[1], acc2); acc2 = fmaf(x2, ws[2], acc2);
                    }
                    if (valid > 3) {
                        const float* ws = sw + 3 * 81 + w_kh_base;
                        acc3 = fmaf(x0, ws[0], acc3); acc3 = fmaf(x1, ws[1], acc3); acc3 = fmaf(x2, ws[2], acc3);
                    }
                    if (valid > 4) {
                        const float* ws = sw + 4 * 81 + w_kh_base;
                        acc4 = fmaf(x0, ws[0], acc4); acc4 = fmaf(x1, ws[1], acc4); acc4 = fmaf(x2, ws[2], acc4);
                    }
                    if (valid > 5) {
                        const float* ws = sw + 5 * 81 + w_kh_base;
                        acc5 = fmaf(x0, ws[0], acc5); acc5 = fmaf(x1, ws[1], acc5); acc5 = fmaf(x2, ws[2], acc5);
                    }
                    if (valid > 6) {
                        const float* ws = sw + 6 * 81 + w_kh_base;
                        acc6 = fmaf(x0, ws[0], acc6); acc6 = fmaf(x1, ws[1], acc6); acc6 = fmaf(x2, ws[2], acc6);
                    }
                    if (valid > 7) {
                        const float* ws = sw + 7 * 81 + w_kh_base;
                        acc7 = fmaf(x0, ws[0], acc7); acc7 = fmaf(x1, ws[1], acc7); acc7 = fmaf(x2, ws[2], acc7);
                    }
                }
            }
        }

        // store
        int64_t out_base = y_index_ncdhw(n, 0, od, oh, ow, Cout, Dout, Hout, Wout);
        int64_t co_stride = (int64_t)out_vox;

        if (valid > 0) y[out_base + (int64_t)(oc0 + 0) * co_stride] = acc0;
        if (valid > 1) y[out_base + (int64_t)(oc0 + 1) * co_stride] = acc1;
        if (valid > 2) y[out_base + (int64_t)(oc0 + 2) * co_stride] = acc2;
        if (valid > 3) y[out_base + (int64_t)(oc0 + 3) * co_stride] = acc3;
        if (valid > 4) y[out_base + (int64_t)(oc0 + 4) * co_stride] = acc4;
        if (valid > 5) y[out_base + (int64_t)(oc0 + 5) * co_stride] = acc5;
        if (valid > 6) y[out_base + (int64_t)(oc0 + 6) * co_stride] = acc6;
        if (valid > 7) y[out_base + (int64_t)(oc0 + 7) * co_stride] = acc7;
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor x, torch::Tensor w, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,Cin,D,H,W)");
    TORCH_CHECK(w.dim() == 5, "w must be 5D (Cout,Cin,K,K,K)");
    TORCH_CHECK(x.size(1) == w.size(1), "Cin mismatch");
    TORCH_CHECK(w.size(2) == w.size(3) && w.size(3) == w.size(4), "Kernel must be cubic (K,K,K)");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();

    const int N    = (int)x.size(0);
    const int Cin  = (int)x.size(1);
    const int D    = (int)x.size(2);
    const int H    = (int)x.size(3);
    const int W    = (int)x.size(4);

    const int Cout = (int)w.size(0);
    const int K    = (int)w.size(2);

    const int s = (int)stride;
    const int p = (int)padding;

    TORCH_CHECK(s > 0, "stride must be > 0");
    TORCH_CHECK(p >= 0, "padding must be >= 0");

    const int Dout = (D + 2*p - K) / s + 1;
    const int Hout = (H + 2*p - K) / s + 1;
    const int Wout = (W + 2*p - K) / s + 1;
    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "Invalid output size");

    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x.options());

    const float* x_ptr = (const float*)x.data_ptr<float>();
    const float* w_ptr = (const float*)w.data_ptr<float>();
    float* y_ptr = (float*)y.data_ptr<float>();

    bool fast = (Cin == 3) && (K == 3) && (s == 1) && (p == 0) && (Dout == D - 2) && (Hout == H - 2) && (Wout == W - 2);

    if (fast) {
        int threads = 64;
        int64_t voxels = (int64_t)N * (int64_t)Dout * (int64_t)Hout * (int64_t)Wout;

        // blocks_x: enough to cover voxels, capped; grid-stride loop inside handles remainder
        int blocks_x = (int)div_up_i64(voxels, threads);
        if (blocks_x > 65535) blocks_x = 65535;

        int blocks_y = (Cout + 7) / 8;
        dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, 1);

        conv3d_fwd_k3_cin3_s1p0_oc8_wsmem_kernel<<<grid, threads>>>(
            x_ptr, w_ptr, y_ptr, N, D, H, W, Cout
        );
        return y;
    }

    // Generic fallback
    int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)Dout * (int64_t)Hout * (int64_t)Wout;
    int threads = 256;
    int blocks = (int)div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;

    conv3d_fwd_ncdhw_f32_generic_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, y_ptr,
        N, Cin, D, H, W,
        Cout, K,
        s, p,
        Dout, Hout, Wout
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv3d_forward_cuda(torch::Tensor x, torch::Tensor w, int64_t stride, int64_t padding);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_square_opt_v8",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3d_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Conv3d forward using custom CUDA kernel.

    Assumptions:
      - groups=1, dilation=1, bias=False
      - cubic kernel_size (int)
      - float32 CUDA, NCDHW contiguous
      - stride/padding supported (ints)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if bias:
            raise ValueError("Custom conv3d_forward_cuda does not support bias=True")
        if groups != 1:
            raise ValueError("Custom conv3d_forward_cuda does not support groups != 1")
        if dilation != 1:
            raise ValueError("Custom conv3d_forward_cuda does not support dilation != 1")
        if not isinstance(kernel_size, int):
            raise ValueError("kernel_size must be an int for cubic kernels")

        self.stride = int(stride)
        self.padding = int(padding)
        self.custom_ops_lib = custom_ops_lib

        ref = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.weight = nn.Parameter(ref.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()

        w = self.weight
        if not w.is_cuda:
            w = w.cuda()
        if w.dtype != torch.float32:
            w = w.float()

        return self.custom_ops_lib.conv3d_forward_cuda(x, w, self.stride, self.padding)