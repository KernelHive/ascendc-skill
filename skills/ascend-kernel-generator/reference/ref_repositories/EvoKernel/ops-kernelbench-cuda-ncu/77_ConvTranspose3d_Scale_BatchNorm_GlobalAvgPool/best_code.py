import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# v2 improvements over current baseline:
#  - ConvT gather: 2-wide in W per thread (ILP) while preserving coalesced stores
#    (avoid failed 4-wide strided store pattern)
#  - No alignment-branch vector loads (avoid failed float4 alignment branch pattern)
#  - Optional constant-memory caching for bias / BN gamma / BN beta (safe fallback)
#  - Heuristic 128 vs 256 thread dispatch for convT to balance regs/occupancy
#  - --use_fast_math enabled (FP32), keep numerics stable (no atomics/reordering)
# ============================================================

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// ------------------------------
// Constant memory for per-channel vectors (optional)
// ------------------------------
#ifndef CMAX
#define CMAX 2048
#endif

__constant__ float c_bias[CMAX];
__constant__ float c_gamma[CMAX];
__constant__ float c_beta[CMAX];

static inline void copy_to_const_if_fit(const float* hptr_bias, const float* hptr_gamma, const float* hptr_beta, int C, cudaStream_t stream) {
    if (C <= CMAX) {
        cudaMemcpyToSymbolAsync(c_bias,  hptr_bias,  sizeof(float) * C, 0, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyToSymbolAsync(c_gamma, hptr_gamma, sizeof(float) * C, 0, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyToSymbolAsync(c_beta,  hptr_beta,  sizeof(float) * C, 0, cudaMemcpyDeviceToDevice, stream);
    }
}

// ------------------------------
// 1) ConvTranspose3d forward (GATHER) specialized to stride=1, pad=0, dilation=1.
// v2: compute 2 output W positions per thread to increase ILP without harming store coalescing.
// Mapping: linear index covers pairs along W (ow, ow+1).
// ------------------------------
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void convT3d_gather_s1_p0_w2_kernel(
    const float* __restrict__ x, // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w, // [Cin,Cout,K,K,K]
    float* __restrict__ y,       // [N,Cout,Dout,Hout,Wout]
    int N, int Cin, int Cout,
    int Din, int Hin, int Win,
    int K,
    int Dout, int Hout, int Wout
){
    // total pairs along W
    int64_t Wpairs = (Wout + 1) >> 1;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Cout * (int64_t)Dout * Hout * Wpairs;
    if (idx >= total) return;

    int64_t t = idx;
    int wp = (int)(t % Wpairs); t /= Wpairs;
    int oh = (int)(t % Hout);   t /= Hout;
    int od = (int)(t % Dout);   t /= Dout;
    int co = (int)(t % Cout);   t /= Cout;
    int n  = (int)t;

    int ow0 = wp << 1;
    int ow1 = ow0 + 1;
    bool has1 = (ow1 < Wout);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int64_t in_hw  = (int64_t)Hin * Win;
    int64_t in_dhw = (int64_t)Din * in_hw;
    int64_t x_n_base = (int64_t)n * (int64_t)Cin * in_dhw;

    int64_t K3 = (int64_t)K * K * K;
    int64_t w_ci_stride = (int64_t)Cout * K3;
    int64_t w_co_stride = K3;

    #pragma unroll 1
    for (int ci = 0; ci < Cin; ++ci) {
        int64_t w_base_ci_co = (int64_t)ci * w_ci_stride + (int64_t)co * w_co_stride;

        #pragma unroll 1
        for (int kd = 0; kd < K; ++kd) {
            int id = od - kd;
            if ((unsigned)id >= (unsigned)Din) continue;

            #pragma unroll 1
            for (int kh = 0; kh < K; ++kh) {
                int ih = oh - kh;
                if ((unsigned)ih >= (unsigned)Hin) continue;

                int64_t x_base_ci = x_n_base + (int64_t)ci * in_dhw + (int64_t)id * in_hw + (int64_t)ih * Win;

                #pragma unroll 1
                for (int kw = 0; kw < K; ++kw) {
                    int iw0 = ow0 - kw;
                    int iw1 = ow1 - kw;

                    int64_t kidx = ((int64_t)kd * K + kh) * (int64_t)K + kw;
                    float wv = __ldg(w + w_base_ci_co + kidx);

                    if ((unsigned)iw0 < (unsigned)Win) {
                        float xv0 = __ldg(x + x_base_ci + iw0);
                        acc0 = fmaf(xv0, wv, acc0);
                    }
                    if (has1 && (unsigned)iw1 < (unsigned)Win) {
                        float xv1 = __ldg(x + x_base_ci + iw1);
                        acc1 = fmaf(xv1, wv, acc1);
                    }
                }
            }
        }
    }

    int64_t out_hw  = (int64_t)Hout * Wout;
    int64_t out_dhw = (int64_t)Dout * out_hw;
    int64_t y_base = (((int64_t)n * Cout + co) * (int64_t)Dout + od) * out_hw + (int64_t)oh * Wout + ow0;

    // coalesced: threads with consecutive wp write consecutive ow0 positions
    y[y_base] = acc0;
    if (has1) y[y_base + 1] = acc1;
}

// ------------------------------
// 2) BN stats over (y + bias) * scale, mean/var per channel
// (kept similar; optionally read bias from constant memory)
// ------------------------------
__global__ __launch_bounds__(256, 2)
void bn_stats_from_y_warp_kernel(
    const float* __restrict__ y,    // [N,C,D,H,W]
    const float* __restrict__ bias, // [C]
    float* __restrict__ mean,       // [C]
    float* __restrict__ var,        // [C]
    int N, int C, int D, int H, int W,
    float scale,
    int use_const
){
    int c = (int)blockIdx.x;
    if (c >= C) return;

    int64_t HW = (int64_t)D * H * W;
    int64_t M = (int64_t)N * HW;

    float b = use_const ? c_bias[c] : __ldg(bias + c);

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int64_t i = (int64_t)threadIdx.x; i < M; i += (int64_t)blockDim.x) {
        int n = (int)(i / HW);
        int64_t s = i - (int64_t)n * HW;
        int64_t off = ((int64_t)n * C + c) * HW + s;
        float v = __ldg(y + off);
        float vs = (v + b) * scale;
        sum += vs;
        sumsq = fmaf(vs, vs, sumsq);
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    __shared__ float warp_sum[8];
    __shared__ float warp_sumsq[8];
    if (lane == 0) {
        warp_sum[warp] = sum;
        warp_sumsq[warp] = sumsq;
    }
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < (blockDim.x >> 5)) ? warp_sum[lane] : 0.0f;
        float bsumsq = (lane < (blockDim.x >> 5)) ? warp_sumsq[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        bsumsq = warp_reduce_sum(bsumsq);
        if (lane == 0) {
            float m = bsum / (float)M;
            float v = bsumsq / (float)M - m * m;
            mean[c] = m;
            var[c] = v;
        }
    }
}

// ------------------------------
// 3) Fused BN affine + GlobalAvgPool into [N,C]
// Optionally read bias/gamma/beta from constant memory
// ------------------------------
__global__ __launch_bounds__(256, 2)
void bn_affine_gap_fused_kernel(
    const float* __restrict__ y,     // [N,C,D,H,W]
    const float* __restrict__ bias,  // [C]
    const float* __restrict__ mean,  // [C]
    const float* __restrict__ var,   // [C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ out,         // [N,C]
    int N, int C, int D, int H, int W,
    float scale,
    float eps,
    int use_const
){
    int c = (int)blockIdx.x;
    int n = (int)blockIdx.y;
    if (c >= C || n >= N) return;

    int64_t HW = (int64_t)D * H * W;
    int64_t base = ((int64_t)n * C + c) * HW;

    float b  = use_const ? c_bias[c] : __ldg(bias + c);
    float m  = __ldg(mean + c);
    float vv = __ldg(var + c);
    float inv = rsqrtf(vv + eps);
    float g  = use_const ? c_gamma[c] : __ldg(gamma + c);
    float be = use_const ? c_beta[c]  : __ldg(beta + c);

    float acc = 0.0f;
    for (int64_t i = (int64_t)threadIdx.x; i < HW; i += (int64_t)blockDim.x) {
        float v = __ldg(y + base + i);
        float vs = (v + b) * scale;
        float yn = (vs - m) * inv;
        float outv = fmaf(yn, g, be);
        acc += outv;
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    acc = warp_reduce_sum(acc);

    __shared__ float warp_acc[8];
    if (lane == 0) warp_acc[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        float bacc = (lane < (blockDim.x >> 5)) ? warp_acc[lane] : 0.0f;
        bacc = warp_reduce_sum(bacc);
        if (lane == 0) {
            out[(int64_t)n * C + c] = bacc * (1.0f / (float)HW);
        }
    }
}

// ------------------------------
// C++/CUDA entrypoint
// ------------------------------
torch::Tensor conv_transpose3d_scale_batch_norm_global_avg_pool_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    double scale_factor,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
){
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(b);
    CHECK_CUDA(bn_weight); CHECK_CUDA(bn_bias);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(b);
    CHECK_CONTIGUOUS(bn_weight); CHECK_CONTIGUOUS(bn_bias);
    CHECK_F32(x); CHECK_F32(w); CHECK_F32(b);
    CHECK_F32(bn_weight); CHECK_F32(bn_bias);

    TORCH_CHECK(x.dim() == 5, "x must be [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be [Cin,Cout,K,K,K]");
    TORCH_CHECK(b.dim() == 1, "b must be [Cout]");
    TORCH_CHECK(bn_weight.dim() == 1 && bn_bias.dim() == 1, "bn params must be [Cout]");
    TORCH_CHECK(w.size(2) == w.size(3) && w.size(3) == w.size(4), "kernel must be cubic KxKxK");

    int64_t N   = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Din = x.size(2);
    int64_t Hin = x.size(3);
    int64_t Win = x.size(4);

    TORCH_CHECK(w.size(0) == Cin, "w.size(0) must match Cin");
    int64_t Cout = w.size(1);
    int64_t K    = w.size(2);

    TORCH_CHECK(b.numel() == Cout, "bias must have Cout elements");
    TORCH_CHECK(bn_weight.numel() == Cout && bn_bias.numel() == Cout, "bn params must have Cout elements");

    int64_t Dout = Din + K - 1;
    int64_t Hout = Hin + K - 1;
    int64_t Wout = Win + K - 1;

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();

    int use_const = (Cout <= CMAX) ? 1 : 0;
    if (use_const) {
        copy_to_const_if_fit(b.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), (int)Cout, stream);
    }

    // 1) convT gather output (W2 ILP)
    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x.options());
    {
        int64_t Wpairs = (Wout + 1) >> 1;
        int64_t total_pairs = (int64_t)N * Cout * Dout * Hout * Wpairs;

        // Heuristic: smaller CTA can reduce regs and raise occupancy for heavy kernels
        int64_t work = Cin * (int64_t)K * K * K;
        if (work >= 8192) {
            const int threads = 256;
            int blocks = (int)((total_pairs + threads - 1) / threads);
            convT3d_gather_s1_p0_w2_kernel<256><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Cout,
                (int)Din, (int)Hin, (int)Win,
                (int)K,
                (int)Dout, (int)Hout, (int)Wout
            );
        } else {
            const int threads = 128;
            int blocks = (int)((total_pairs + threads - 1) / threads);
            convT3d_gather_s1_p0_w2_kernel<128><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Cout,
                (int)Din, (int)Hin, (int)Win,
                (int)K,
                (int)Dout, (int)Hout, (int)Wout
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 2) BN stats computed from (y + bias) * scale
    auto mean = torch::empty({Cout}, x.options());
    auto var  = torch::empty({Cout}, x.options());
    {
        const int threads = 256;
        dim3 grid((unsigned)Cout);
        bn_stats_from_y_warp_kernel<<<grid, threads, 0, stream>>>(
            y.data_ptr<float>(),
            b.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            (int)N, (int)Cout, (int)Dout, (int)Hout, (int)Wout,
            (float)scale_factor,
            use_const
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 3) Fused BN affine + GAP directly into [N,C]
    auto out_nc = torch::empty({N, Cout}, x.options());
    {
        const int threads = 256;
        dim3 grid((unsigned)Cout, (unsigned)N);
        bn_affine_gap_fused_kernel<<<grid, threads, 0, stream>>>(
            y.data_ptr<float>(),
            b.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            out_nc.data_ptr<float>(),
            (int)N, (int)Cout, (int)Dout, (int)Hout, (int)Wout,
            (float)scale_factor,
            (float)eps,
            use_const
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out_nc.view({N, Cout, 1, 1, 1});
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose3d_scale_batch_norm_global_avg_pool_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    double scale_factor,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convT3d_scale_bn_gap_gather_fused_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_scale_batch_norm_global_avg_pool_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Custom CUDA replacement for:
      ConvTranspose3d(stride=1,padding=0) -> scale -> BatchNorm3d (training stats) -> GlobalAvgPool3d

    Constraints:
      - stride=1, padding=0, dilation=1, output_padding=0, groups=1, bias=True
      - cubic kernel_size
      - float32 CUDA tensors, contiguous
      - BatchNorm uses batch statistics (training-style). Running stats are not updated.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)
        self.eps = float(eps)
        self.momentum = float(momentum)  # signature compatibility only

        w = torch.empty(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))

        self.bn_weight = nn.Parameter(torch.ones(self.out_channels, dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew supports CUDA tensors only")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        dev = x.device
        w = self.weight if self.weight.device == dev else self.weight.to(dev)
        b = self.bias if self.bias.device == dev else self.bias.to(dev)
        gw = self.bn_weight if self.bn_weight.device == dev else self.bn_weight.to(dev)
        gb = self.bn_bias if self.bn_bias.device == dev else self.bn_bias.to(dev)

        return self.custom_ops.conv_transpose3d_scale_batch_norm_global_avg_pool_cuda(
            x,
            w.contiguous(),
            b.contiguous(),
            float(self.scale_factor),
            gw.contiguous(),
            gb.contiguous(),
            float(self.eps),
        )