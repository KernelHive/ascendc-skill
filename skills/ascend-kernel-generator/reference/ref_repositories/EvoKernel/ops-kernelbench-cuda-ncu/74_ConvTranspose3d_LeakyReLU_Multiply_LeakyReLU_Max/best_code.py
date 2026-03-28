import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: fuse (leaky_relu -> multiply -> leaky_relu) + maxpool3d(k=2,s=2)
# Improvements over baseline:
#  - each thread computes 2 output W positions (wo, wo+1) when possible (float2 store)
#  - grid-stride loop
#  - __ldg for multiplier
#  - __launch_bounds__ for occupancy tuning
#  - NDHWC (channels-last 3d) fast path kernel
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float leaky_relu_f(float x, float neg_slope) {
    return x >= 0.0f ? x : x * neg_slope;
}

__device__ __forceinline__ float fused_act(float x, float mult, float neg_slope) {
    x = leaky_relu_f(x, neg_slope);
    x = x * mult;
    x = leaky_relu_f(x, neg_slope);
    return x;
}

__device__ __forceinline__ float max8(float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7){
    float m = a0;
    m = fmaxf(m,a1); m = fmaxf(m,a2); m = fmaxf(m,a3);
    m = fmaxf(m,a4); m = fmaxf(m,a5); m = fmaxf(m,a6); m = fmaxf(m,a7);
    return m;
}

// ---------------- NCDHW kernel: each thread computes two Wo outputs when possible ----------------
__global__ __launch_bounds__(256, 2) void fused_ncdhw_w2_kernel(
    const float* __restrict__ x,   // [N,C,D,H,W]
    const float* __restrict__ m,   // [C]
    float* __restrict__ y,         // [N,C,Do,Ho,Wo]
    int N, int C, int D, int H, int W,
    int Do, int Ho, int Wo,
    float neg_slope
) {
    // Pair outputs along Wo: wo_pair in [0, (Wo+1)/2)
    const int Wo2 = (Wo + 1) >> 1;
    int64_t total_pairs = (int64_t)N * (int64_t)C * (int64_t)Do * (int64_t)Ho * (int64_t)Wo2;

    const int64_t HW  = (int64_t)H * (int64_t)W;
    const int64_t DHW = (int64_t)D * HW;

    auto stream_total = total_pairs;
    for (int64_t pair_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pair_idx < stream_total;
         pair_idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = pair_idx;
        int wo2 = (int)(t % Wo2); t /= Wo2;
        int ho  = (int)(t % Ho);  t /= Ho;
        int do_ = (int)(t % Do);  t /= Do;
        int c   = (int)(t % C);   t /= C;
        int n   = (int)t;

        const int wo0 = wo2 * 2;
        const int wo1 = wo0 + 1;
        const bool has1 = (wo1 < Wo);

        const float mult = ldg_f(m + c);

        const int di0 = do_ * 2;
        const int hi0 = ho  * 2;
        const int wi0_0 = wo0 * 2;
        const int wi0_1 = wi0_0 + 2; // for wo1

        const int64_t base_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * DHW;
        const int64_t base_d0 = base_nc + (int64_t)di0 * HW;
        const int64_t base_d1 = base_d0 + HW;

        const int64_t base_h00 = base_d0 + (int64_t)hi0 * (int64_t)W;
        const int64_t base_h01 = base_h00 + (int64_t)W;

        const int64_t base_h10 = base_d1 + (int64_t)hi0 * (int64_t)W;
        const int64_t base_h11 = base_h10 + (int64_t)W;

        // wo0 window (wi0_0 .. wi0_0+1)
        const int64_t b0 = base_h00 + (int64_t)wi0_0;
        float v0 = fused_act(x[b0 + 0], mult, neg_slope);
        float v1 = fused_act(x[b0 + 1], mult, neg_slope);

        const int64_t b1 = base_h01 + (int64_t)wi0_0;
        float v2 = fused_act(x[b1 + 0], mult, neg_slope);
        float v3 = fused_act(x[b1 + 1], mult, neg_slope);

        const int64_t b2 = base_h10 + (int64_t)wi0_0;
        float v4 = fused_act(x[b2 + 0], mult, neg_slope);
        float v5 = fused_act(x[b2 + 1], mult, neg_slope);

        const int64_t b3 = base_h11 + (int64_t)wi0_0;
        float v6 = fused_act(x[b3 + 0], mult, neg_slope);
        float v7 = fused_act(x[b3 + 1], mult, neg_slope);

        float out0 = max8(v0,v1,v2,v3,v4,v5,v6,v7);

        float out1 = 0.0f;
        if (has1) {
            const int64_t c0 = base_h00 + (int64_t)wi0_1;
            float u0 = fused_act(x[c0 + 0], mult, neg_slope);
            float u1 = fused_act(x[c0 + 1], mult, neg_slope);

            const int64_t c1 = base_h01 + (int64_t)wi0_1;
            float u2 = fused_act(x[c1 + 0], mult, neg_slope);
            float u3 = fused_act(x[c1 + 1], mult, neg_slope);

            const int64_t c2 = base_h10 + (int64_t)wi0_1;
            float u4 = fused_act(x[c2 + 0], mult, neg_slope);
            float u5 = fused_act(x[c2 + 1], mult, neg_slope);

            const int64_t c3 = base_h11 + (int64_t)wi0_1;
            float u6 = fused_act(x[c3 + 0], mult, neg_slope);
            float u7 = fused_act(x[c3 + 1], mult, neg_slope);

            out1 = max8(u0,u1,u2,u3,u4,u5,u6,u7);
        }

        // Write out: y layout contiguous in Wo, so we can float2-store when has1
        int64_t y_base = ((((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Do + (int64_t)do_) * (int64_t)Ho + (int64_t)ho) * (int64_t)Wo + (int64_t)wo0;

        if (has1) {
            // y_base is 4-byte aligned; float2 store requires 8-byte alignment; guard it.
            if (((y_base & 1LL) == 0)) {
                reinterpret_cast<float2*>(y)[y_base >> 1] = make_float2(out0, out1);
            } else {
                y[y_base] = out0;
                y[y_base + 1] = out1;
            }
        } else {
            y[y_base] = out0;
        }
    }
}

// ---------------- NDHWC kernel: [N,D,H,W,C] contiguous channels-last ----------------
__global__ __launch_bounds__(256, 2) void fused_ndhwc_w2_kernel(
    const float* __restrict__ x,   // [N,D,H,W,C]
    const float* __restrict__ m,   // [C]
    float* __restrict__ y,         // [N,Do,Ho,Wo,C] (channels-last)
    int N, int C, int D, int H, int W,
    int Do, int Ho, int Wo,
    float neg_slope
) {
    const int Wo2 = (Wo + 1) >> 1;
    int64_t total_pairs = (int64_t)N * (int64_t)Do * (int64_t)Ho * (int64_t)Wo2 * (int64_t)C;

    const int64_t WC = (int64_t)W * (int64_t)C;
    const int64_t HWC = (int64_t)H * WC;
    const int64_t DHWC = (int64_t)D * HWC;

    for (int64_t pair_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pair_idx < total_pairs;
         pair_idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = pair_idx;
        int c   = (int)(t % C);   t /= C;
        int wo2 = (int)(t % Wo2); t /= Wo2;
        int ho  = (int)(t % Ho);  t /= Ho;
        int do_ = (int)(t % Do);  t /= Do;
        int n   = (int)t;

        const int wo0 = wo2 * 2;
        const int wo1 = wo0 + 1;
        const bool has1 = (wo1 < Wo);

        const float mult = ldg_f(m + c);

        const int di0 = do_ * 2;
        const int hi0 = ho  * 2;
        const int wi0_0 = wo0 * 2;
        const int wi0_1 = wi0_0 + 2;

        const int64_t n_base = (int64_t)n * DHWC;

        auto load_act = [&](int d, int h, int w) -> float {
            int64_t idx = n_base + (int64_t)d * HWC + (int64_t)h * WC + (int64_t)w * (int64_t)C + (int64_t)c;
            return fused_act(x[idx], mult, neg_slope);
        };

        float v0 = load_act(di0 + 0, hi0 + 0, wi0_0 + 0);
        float v1 = load_act(di0 + 0, hi0 + 0, wi0_0 + 1);
        float v2 = load_act(di0 + 0, hi0 + 1, wi0_0 + 0);
        float v3 = load_act(di0 + 0, hi0 + 1, wi0_0 + 1);
        float v4 = load_act(di0 + 1, hi0 + 0, wi0_0 + 0);
        float v5 = load_act(di0 + 1, hi0 + 0, wi0_0 + 1);
        float v6 = load_act(di0 + 1, hi0 + 1, wi0_0 + 0);
        float v7 = load_act(di0 + 1, hi0 + 1, wi0_0 + 1);

        float out0 = max8(v0,v1,v2,v3,v4,v5,v6,v7);

        float out1 = 0.0f;
        if (has1) {
            float u0 = load_act(di0 + 0, hi0 + 0, wi0_1 + 0);
            float u1 = load_act(di0 + 0, hi0 + 0, wi0_1 + 1);
            float u2 = load_act(di0 + 0, hi0 + 1, wi0_1 + 0);
            float u3 = load_act(di0 + 0, hi0 + 1, wi0_1 + 1);
            float u4 = load_act(di0 + 1, hi0 + 0, wi0_1 + 0);
            float u5 = load_act(di0 + 1, hi0 + 0, wi0_1 + 1);
            float u6 = load_act(di0 + 1, hi0 + 1, wi0_1 + 0);
            float u7 = load_act(di0 + 1, hi0 + 1, wi0_1 + 1);

            out1 = max8(u0,u1,u2,u3,u4,u5,u6,u7);
        }

        // y is [N,Do,Ho,Wo,C] contiguous (channels-last)
        int64_t y_base0 = (((((int64_t)n * (int64_t)Do + (int64_t)do_) * (int64_t)Ho + (int64_t)ho) * (int64_t)Wo + (int64_t)wo0) * (int64_t)C) + (int64_t)c;
        y[y_base0] = out0;
        if (has1) {
            y[y_base0 + (int64_t)C] = out1; // next w has stride C
        }
    }
}

torch::Tensor leaky_relu_multiply_leaky_relu_maxpool2_cuda(
    torch::Tensor x,
    torch::Tensor multiplier,   // [C]
    double negative_slope
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "multiplier must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(multiplier.dtype() == torch::kFloat32, "multiplier must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D");

    c10::cuda::CUDAGuard device_guard(x.device());

    auto m_c = multiplier.contiguous();
    TORCH_CHECK(m_c.dim() == 1, "multiplier must be 1D [C]");

    const float neg = (float)negative_slope;

    // Detect channels-last 3D contiguous: [N,D,H,W,C]
    const bool is_cl = x.is_contiguous(at::MemoryFormat::ChannelsLast3d);

    if (!is_cl) {
        auto x_c = x.contiguous(); // NCDHW
        int N = (int)x_c.size(0);
        int C = (int)x_c.size(1);
        int D = (int)x_c.size(2);
        int H = (int)x_c.size(3);
        int W = (int)x_c.size(4);

        TORCH_CHECK((int)m_c.size(0) == C, "multiplier size must match channel dimension");
        TORCH_CHECK(D >= 2 && H >= 2 && W >= 2, "Input spatial dims must be >= 2 for pool2");

        int Do = D / 2;
        int Ho = H / 2;
        int Wo = W / 2;

        auto y = torch::empty({N, C, Do, Ho, Wo}, x_c.options());

        const int threads = 256;
        const int64_t Wo2 = (Wo + 1) >> 1;
        int64_t total_pairs = (int64_t)N * (int64_t)C * (int64_t)Do * (int64_t)Ho * Wo2;

        int blocks = (int)((total_pairs + threads - 1) / threads);
        if (blocks > 32768) blocks = 32768;
        if (blocks < 1) blocks = 1;

        auto stream = at::cuda::getDefaultCUDAStream();
        fused_ncdhw_w2_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)x_c.data_ptr<float>(),
            (const float*)m_c.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, D, H, W,
            Do, Ho, Wo,
            neg
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    } else {
        // channels-last 3D: treat as NDHWC logical
        auto x_cl = x.contiguous(at::MemoryFormat::ChannelsLast3d);
        // x_cl sizes still report [N,C,D,H,W], but strides correspond to channels-last.
        // We will reinterpret indexing as [N,D,H,W,C] via strides check and pointer math based on sizes.
        int N = (int)x_cl.size(0);
        int C = (int)x_cl.size(1);
        int D = (int)x_cl.size(2);
        int H = (int)x_cl.size(3);
        int W = (int)x_cl.size(4);

        TORCH_CHECK((int)m_c.size(0) == C, "multiplier size must match channel dimension");
        TORCH_CHECK(D >= 2 && H >= 2 && W >= 2, "Input spatial dims must be >= 2 for pool2");

        int Do = D / 2;
        int Ho = H / 2;
        int Wo = W / 2;

        // Create output as channels-last to preserve fast path end-to-end
        auto y = torch::empty({N, C, Do, Ho, Wo}, x_cl.options().memory_format(at::MemoryFormat::ChannelsLast3d));

        const int threads = 256;
        const int64_t Wo2 = (Wo + 1) >> 1;
        int64_t total_pairs = (int64_t)N * (int64_t)Do * (int64_t)Ho * Wo2 * (int64_t)C;

        int blocks = (int)((total_pairs + threads - 1) / threads);
        if (blocks > 32768) blocks = 32768;
        if (blocks < 1) blocks = 1;

        // For the NDHWC kernel, we pass x/y pointers but index them as if contiguous NDHWC.
        // This is valid because channels-last 3d contiguous corresponds to NDHWC contiguous in memory.
        auto stream = at::cuda::getDefaultCUDAStream();
        fused_ndhwc_w2_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)x_cl.data_ptr<float>(),
            (const float*)m_c.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, D, H, W,
            Do, Ho, Wo,
            neg
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor leaky_relu_multiply_leaky_relu_maxpool2_cuda(
    torch::Tensor x,
    torch::Tensor multiplier,
    double negative_slope
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_fused_leaky_mul_leaky_maxpool2_v2_w2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["leaky_relu_multiply_leaky_relu_maxpool2_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) -> fused CUDA: LeakyReLU -> multiply -> LeakyReLU -> MaxPool3d(k=2,s=2)

    Constraints:
      - fused op supports float32 CUDA
      - multiplier broadcast expected as (Cout,1,1,1) or flattenable to [Cout]
      - maxpool is fixed to kernel=2, stride=2, padding=0, ceil_mode=False
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape, dtype=torch.float32))
        self.negative_slope = 0.2
        self.max_pool = nn.MaxPool3d(kernel_size=2)  # unused (fused)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        m = self.multiplier
        if m.dim() == 4 and m.size(1) == 1 and m.size(2) == 1 and m.size(3) == 1:
            m1 = m.view(m.size(0))
        else:
            m1 = m.view(-1)

        x = self.custom_ops.leaky_relu_multiply_leaky_relu_maxpool2_cuda(
            x, m1, float(self.negative_slope)
        )
        return x