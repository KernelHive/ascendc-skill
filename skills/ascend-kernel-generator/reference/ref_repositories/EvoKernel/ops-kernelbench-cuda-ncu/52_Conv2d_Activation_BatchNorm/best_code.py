import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_4D(x) TORCH_CHECK((x).dim() == 4, #x " must be 4D NCHW")

static inline __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
static inline __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
static inline __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    float out = 0.0f;
    if (wid == 0) {
        out = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

__inline__ __device__ bool is_aligned_16(const void* p) { return (((uintptr_t)p) & 0xF) == 0; }

// stable softplus and activation: y = x * tanh(softplus(x))
__device__ __forceinline__ float softplus_stable(float x) {
    float ax = fabsf(x);
    return log1pf(expf(-ax)) + fmaxf(x, 0.0f);
}
__device__ __forceinline__ float act_x_tanh_softplus(float x) {
    return x * tanhf(softplus_stable(x));
}

// ---------------- Conv kernel: conv3x3 valid + activation into act ----------------
// Grid: dim3(grid_x tiles over NHW, grid_y over Cout)
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void conv3x3_act_fwd_tiled(
    const float* __restrict__ x,      // [N,Cin,Hin,Win]
    const float* __restrict__ w,      // [Cout,Cin,3,3] contiguous
    const float* __restrict__ b,      // [Cout]
    float* __restrict__ act,          // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout
) {
    int oc = (int)blockIdx.y;
    if (oc >= Cout) return;

    int64_t HW_in  = (int64_t)Hin * (int64_t)Win;
    int64_t HW_out = (int64_t)Hout * (int64_t)Wout;
    int64_t strideN_in  = (int64_t)Cin  * HW_in;
    int64_t strideN_out = (int64_t)Cout * HW_out;

    int64_t tile = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x * (int64_t)blockDim.x;

    float bias = b[oc];
    int64_t w_oc_base = ((int64_t)oc * (int64_t)Cin) * 9;

    for (int64_t idx = tile; idx < (int64_t)N * HW_out; idx += step) {
        int n = (int)(idx / HW_out);
        int64_t hw = idx - (int64_t)n * HW_out;
        int oh = (int)(hw / Wout);
        int ow = (int)(hw - (int64_t)oh * Wout);

        int iy0 = oh;
        int ix0 = ow;

        float acc = bias;
        int64_t x_n_base = (int64_t)n * strideN_in;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            int64_t x_ic_base = x_n_base + (int64_t)ic * HW_in;
            int64_t w_ic_base = w_oc_base + (int64_t)ic * 9;

            int64_t row0 = x_ic_base + (int64_t)(iy0 + 0) * Win + (ix0 + 0);
            int64_t row1 = x_ic_base + (int64_t)(iy0 + 1) * Win + (ix0 + 0);
            int64_t row2 = x_ic_base + (int64_t)(iy0 + 2) * Win + (ix0 + 0);

            float x00 = x[row0 + 0]; float x01 = x[row0 + 1]; float x02 = x[row0 + 2];
            float x10 = x[row1 + 0]; float x11 = x[row1 + 1]; float x12 = x[row1 + 2];
            float x20 = x[row2 + 0]; float x21 = x[row2 + 1]; float x22 = x[row2 + 2];

            // weights: contiguous and reused; __ldg for read-only cache
            float w00 = __ldg(w + w_ic_base + 0); float w01 = __ldg(w + w_ic_base + 1); float w02 = __ldg(w + w_ic_base + 2);
            float w10 = __ldg(w + w_ic_base + 3); float w11 = __ldg(w + w_ic_base + 4); float w12 = __ldg(w + w_ic_base + 5);
            float w20 = __ldg(w + w_ic_base + 6); float w21 = __ldg(w + w_ic_base + 7); float w22 = __ldg(w + w_ic_base + 8);

            acc = fmaf(x00, w00, acc); acc = fmaf(x01, w01, acc); acc = fmaf(x02, w02, acc);
            acc = fmaf(x10, w10, acc); acc = fmaf(x11, w11, acc); acc = fmaf(x12, w12, acc);
            acc = fmaf(x20, w20, acc); acc = fmaf(x21, w21, acc); acc = fmaf(x22, w22, acc);
        }

        float a = act_x_tanh_softplus(acc);
        int64_t out_off = (int64_t)n * strideN_out + (int64_t)oc * HW_out + hw;
        act[out_off] = a;
    }
}

// ---------------- BN train forward (stats + apply) using Welford ----------------
// One block computes one channel c, persistent over channels.
__device__ __forceinline__ void welford_update(float x, float &mean, float &m2, float &count) {
    count += 1.0f;
    float delta = x - mean;
    mean += delta / count;
    float delta2 = x - mean;
    m2 += delta * delta2;
}

__device__ __forceinline__ void welford_combine(
    float mean_b, float m2_b, float count_b,
    float &mean_a, float &m2_a, float &count_a
) {
    if (count_b == 0.0f) return;
    if (count_a == 0.0f) { mean_a = mean_b; m2_a = m2_b; count_a = count_b; return; }
    float delta = mean_b - mean_a;
    float count = count_a + count_b;
    mean_a = mean_a + delta * (count_b / count);
    m2_a = m2_a + m2_b + delta * delta * (count_a * count_b / count);
    count_a = count;
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void bn2d_train_fwd_welford_fused(
    const float* __restrict__ act,      // [N,C,H,W]
    const float* __restrict__ bn_w,     // [C]
    const float* __restrict__ bn_b,     // [C]
    float* __restrict__ y,             // [N,C,H,W]
    int N, int C, int H, int W,
    float eps
) {
    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t strideN = (int64_t)C * HW;

    for (int c = (int)blockIdx.x; c < C; c += (int)gridDim.x) {
        float gamma = __ldg(bn_w + c);
        float beta  = __ldg(bn_b + c);

        // per-thread welford partials
        float mean = 0.0f;
        float m2 = 0.0f;
        float count = 0.0f;

        // stats pass: prefer float4
        for (int n = 0; n < N; n++) {
            const float* a_nc = act + (int64_t)n * strideN + (int64_t)c * HW;
            bool vec_ok = (HW % 4 == 0) && is_aligned_16(a_nc);
            if (vec_ok) {
                const float4* a4 = reinterpret_cast<const float4*>(a_nc);
                int HW4 = (int)(HW >> 2);
                for (int i = threadIdx.x; i < HW4; i += blockDim.x) {
                    float4 v = a4[i];
                    welford_update(v.x, mean, m2, count);
                    welford_update(v.y, mean, m2, count);
                    welford_update(v.z, mean, m2, count);
                    welford_update(v.w, mean, m2, count);
                }
            } else {
                for (int64_t i = (int64_t)threadIdx.x; i < HW; i += (int64_t)blockDim.x) {
                    float v = a_nc[i];
                    welford_update(v, mean, m2, count);
                }
            }
        }

        // reduce welford across block: do warp-level then block-level combine
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;

        // warp reduce by shuffling: combine pairs
        for (int offset = 16; offset > 0; offset >>= 1) {
            float mean_b  = __shfl_down_sync(0xffffffff, mean,  offset);
            float m2_b    = __shfl_down_sync(0xffffffff, m2,    offset);
            float count_b = __shfl_down_sync(0xffffffff, count, offset);
            welford_combine(mean_b, m2_b, count_b, mean, m2, count);
        }

        __shared__ float sh_mean[32];
        __shared__ float sh_m2[32];
        __shared__ float sh_count[32];

        if (lane == 0) {
            sh_mean[wid] = mean;
            sh_m2[wid] = m2;
            sh_count[wid] = count;
        }
        __syncthreads();

        float mean_all = 0.0f, m2_all = 0.0f, count_all = 0.0f;
        if (wid == 0) {
            // one warp loads warp results
            float mean0  = (threadIdx.x < (blockDim.x >> 5)) ? sh_mean[lane] : 0.0f;
            float m20    = (threadIdx.x < (blockDim.x >> 5)) ? sh_m2[lane]   : 0.0f;
            float count0 = (threadIdx.x < (blockDim.x >> 5)) ? sh_count[lane]: 0.0f;

            mean_all = mean0; m2_all = m20; count_all = count0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                float mean_b  = __shfl_down_sync(0xffffffff, mean_all,  offset);
                float m2_b    = __shfl_down_sync(0xffffffff, m2_all,    offset);
                float count_b = __shfl_down_sync(0xffffffff, count_all, offset);
                welford_combine(mean_b, m2_b, count_b, mean_all, m2_all, count_all);
            }
        }

        __shared__ float sh_final_mean;
        __shared__ float sh_final_invstd;
        if (threadIdx.x == 0) {
            // count_all should be N*HW
            float var = (count_all > 1.0f) ? (m2_all / count_all) : 0.0f;
            var = var > 0.0f ? var : 0.0f;
            sh_final_mean = mean_all;
            sh_final_invstd = rsqrtf(var + eps);
        }
        __syncthreads();

        float mu = sh_final_mean;
        float invstd = sh_final_invstd;

        // apply pass: float4 when possible
        for (int n = 0; n < N; n++) {
            const float* a_nc = act + (int64_t)n * strideN + (int64_t)c * HW;
            float* y_nc       = y   + (int64_t)n * strideN + (int64_t)c * HW;

            bool vec_ok = (HW % 4 == 0) && is_aligned_16(a_nc) && is_aligned_16(y_nc);
            if (vec_ok) {
                const float4* a4 = reinterpret_cast<const float4*>(a_nc);
                float4* y4 = reinterpret_cast<float4*>(y_nc);
                int HW4 = (int)(HW >> 2);
                for (int i = threadIdx.x; i < HW4; i += blockDim.x) {
                    float4 v = a4[i];
                    v.x = (v.x - mu) * invstd * gamma + beta;
                    v.y = (v.y - mu) * invstd * gamma + beta;
                    v.z = (v.z - mu) * invstd * gamma + beta;
                    v.w = (v.w - mu) * invstd * gamma + beta;
                    y4[i] = v;
                }
            } else {
                for (int64_t i = (int64_t)threadIdx.x; i < HW; i += (int64_t)blockDim.x) {
                    float v = a_nc[i];
                    y_nc[i] = (v - mu) * invstd * gamma + beta;
                }
            }
        }
        __syncthreads();
    }
}

torch::Tensor conv2d_act_bn_train_fwd_cuda_v3(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
) {
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(b); CHECK_CUDA(bn_weight); CHECK_CUDA(bn_bias);
    CHECK_FLOAT(x); CHECK_FLOAT(w); CHECK_FLOAT(b); CHECK_FLOAT(bn_weight); CHECK_FLOAT(bn_bias);
    CHECK_4D(x); CHECK_4D(w);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(b); CHECK_CONTIGUOUS(bn_weight); CHECK_CONTIGUOUS(bn_bias);

    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3, "only 3x3 kernel supported");
    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);

    int Cout = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == Cin, "Cin mismatch");
    TORCH_CHECK(b.numel() == Cout, "conv bias must be [Cout]");
    TORCH_CHECK(bn_weight.numel() == Cout && bn_bias.numel() == Cout, "BN weight/bias must be [Cout]");

    int Hout = Hin - 2;
    int Wout = Win - 2;
    TORCH_CHECK(Hout > 0 && Wout > 0, "input too small for valid 3x3 conv");

    auto act = torch::empty({N, Cout, Hout, Wout}, x.options());
    auto y   = torch::empty({N, Cout, Hout, Wout}, x.options());

    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // conv+activation
    constexpr int THREADS_CONV = 128;
    int64_t NHW = (int64_t)N * (int64_t)Hout * (int64_t)Wout;

    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm = prop.multiProcessorCount;

    int grid_x = (int)((NHW + THREADS_CONV - 1) / THREADS_CONV);
    int max_grid_x = sm * 32;
    if (grid_x > max_grid_x) grid_x = max_grid_x;
    if (grid_x < 1) grid_x = 1;

    dim3 gridA((unsigned)grid_x, (unsigned)Cout, 1);
    conv3x3_act_fwd_tiled<THREADS_CONV><<<gridA, THREADS_CONV, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)w.data_ptr<float>(),
        (const float*)b.data_ptr<float>(),
        (float*)act.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // BN stats+apply fused
    constexpr int THREADS_BN = 256;
    int bn_blocks = sm * 6;              // more CTAs to improve scheduling across channels
    if (bn_blocks < 1) bn_blocks = 1;
    if (bn_blocks > Cout) bn_blocks = Cout;

    bn2d_train_fwd_welford_fused<THREADS_BN><<<bn_blocks, THREADS_BN, 0, stream>>>(
        (const float*)act.data_ptr<float>(),
        (const float*)bn_weight.data_ptr<float>(),
        (const float*)bn_bias.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, Cout, Hout, Wout,
        (float)eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv2d_act_bn_train_fwd_cuda_v3(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_conv2d_act_bn_train_ops_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv2d_act_bn_train_fwd_cuda_v3"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Drop-in replacement for:
      conv2d -> (x * tanh(softplus(x))) -> BatchNorm2d

    Optimized CUDA path (training forward only) for:
      conv3x3, stride=1, padding=0, dilation=1, groups=1, bias=True
      BN affine=True
    Falls back to PyTorch otherwise (including eval mode).
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4):
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x

        if not (
            self.conv.kernel_size == (3, 3)
            and self.conv.stride == (1, 1)
            and self.conv.padding == (0, 0)
            and self.conv.dilation == (1, 1)
            and self.conv.groups == 1
            and (self.conv.bias is not None)
        ):
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x

        if (not self.bn.affine) or (self.bn.weight is None) or (self.bn.bias is None):
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        w = self.conv.weight
        b = self.conv.bias
        bn_w = self.bn.weight
        bn_b = self.bn.bias

        if w.device != x.device:
            w = w.to(device=x.device)
        if b.device != x.device:
            b = b.to(device=x.device)
        if bn_w.device != x.device:
            bn_w = bn_w.to(device=x.device)
        if bn_b.device != x.device:
            bn_b = bn_b.to(device=x.device)

        if not w.is_contiguous():
            w = w.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not bn_w.is_contiguous():
            bn_w = bn_w.contiguous()
        if not bn_b.is_contiguous():
            bn_b = bn_b.contiguous()

        y = custom_ops_lib.conv2d_act_bn_train_fwd_cuda_v3(
            x, w, b, bn_w, bn_b, float(self.bn.eps)
        )

        # Preserve running stats update semantics (same approach as baseline: update running stats via PyTorch BN)
        if self.bn.track_running_stats:
            with torch.no_grad():
                z = self.conv(x)
                z = torch.multiply(torch.tanh(F.softplus(z)), z)
                F.batch_norm(
                    z,
                    self.bn.running_mean,
                    self.bn.running_var,
                    weight=None,
                    bias=None,
                    training=True,
                    momentum=self.bn.momentum,
                    eps=self.bn.eps,
                )

        return y