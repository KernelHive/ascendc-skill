import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA extension: fused (x - s1) -> tanh -> ( - s2 ) -> avg_pool2d
# - Special fast path for AvgPool2d(k=2,s=2,p=0,ceil_mode=False,count_include_pad=False)
#   with warp-coalesced output mapping and vectorized (4 outputs/thread) processing when possible.
# - Generic fallback for other parameters.
# ----------------------------

fused_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

static inline int64_t div_up_int64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float fused_act(float v, float sub1, float sub2) {
    // --use_fast_math enabled; tanhf becomes fast approx.
    return tanhf(v - sub1) - sub2;
}

// Use __ldg on pre-Ampere too; compiler will ignore when not beneficial.
__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Specialized kernel: k=2,s=2,p=0
// Warp-coalesced mapping over ow; each warp covers contiguous ow.
// Each thread computes VEC outputs (VEC=4) when possible.
template<int THREADS, int VEC>
__global__ __launch_bounds__(THREADS, 2)
void fused_tanh_sub_avgpool2d_k2s2p0_warpvec_forward(
    const float* __restrict__ x,  // N,C,H,W
    float* __restrict__ y,        // N,C,outH,outW
    int N, int C, int H, int W,
    int outH, int outW,
    float sub1, float sub2
) {
    // Warp mapping:
    // warps_per_block = THREADS / 32
    // global_warp_id enumerates "tiles" of (n,c,oh) and ow segments.
    constexpr int WARP = 32;
    const int lane = threadIdx.x & (WARP - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = THREADS >> 5;

    // We'll traverse ow in contiguous segments of (WARP * VEC) outputs per warp-iteration.
    const int ow_seg = WARP * VEC;

    // Total number of "lines" (n,c,oh)
    const int64_t lines = (int64_t)N * C * outH;

    // For each line, number of segments along ow:
    const int segs_per_line = (outW + ow_seg - 1) / ow_seg;
    const int64_t total_warps = lines * (int64_t)segs_per_line;

    int64_t global_warp = (int64_t)blockIdx.x * warps_per_block + warp_in_block;
    int64_t warp_stride = (int64_t)gridDim.x * warps_per_block;

    for (; global_warp < total_warps; global_warp += warp_stride) {
        int seg = (int)(global_warp % segs_per_line);
        int64_t line = global_warp / segs_per_line;

        int oh = (int)(line % outH);
        int tmp = (int)(line / outH);
        int c = tmp % C;
        int n = tmp / C;

        int ow0 = seg * ow_seg + lane;  // base ow for this lane (then + 32*k)

        // input base for (n,c,oh)
        int ih = oh * 2;
        int64_t base_nc = ((int64_t)n * C + c) * (int64_t)H * (int64_t)W;
        const float* xrow0 = x + base_nc + (int64_t)ih * W;
        const float* xrow1 = xrow0 + W;

        // output base
        float* yline = y + (((int64_t)n * C + c) * (int64_t)outH + oh) * (int64_t)outW;

        // Compute VEC outputs per lane: ow = ow0 + 32*k
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            int ow = ow0 + k * WARP;
            if (ow < outW) {
                int iw = ow * 2;
                // Always in bounds for p=0, k=2, s=2 and floor output size.
                const float* p0 = xrow0 + iw;
                const float* p1 = xrow1 + iw;

                // Load 2x2 window. Use float2 to reduce instruction count; alignment is usually OK for contiguous.
                // Safe even if unaligned on modern GPUs; fallback not needed as we remain within bounds.
                float2 r0 = *reinterpret_cast<const float2*>(p0);
                float2 r1 = *reinterpret_cast<const float2*>(p1);

                float a00 = fused_act(r0.x, sub1, sub2);
                float a01 = fused_act(r0.y, sub1, sub2);
                float a10 = fused_act(r1.x, sub1, sub2);
                float a11 = fused_act(r1.y, sub1, sub2);

                yline[ow] = 0.25f * (a00 + a01 + a10 + a11);
            }
        }
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_tanh_sub_avgpool2d_generic_forward_kernel_1d(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    float sub1, float sub2
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * outH * outW;

    for (int64_t idx = tid; idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        int ow = (int)(idx % outW);
        int tmp = (int)(idx / outW);
        int oh = (int)(tmp % outH);
        tmp /= outH;
        int c = (int)(tmp % C);
        int n = (int)(tmp / C);

        int hstart = oh * sH - pH;
        int wstart = ow * sW - pW;
        int hend = hstart + kH;
        int wend = wstart + kW;

        int h0 = hstart < 0 ? 0 : hstart;
        int w0 = wstart < 0 ? 0 : wstart;
        int h1 = hend > H ? H : hend;
        int w1 = wend > W ? W : wend;

        float sum = 0.0f;
        int count = 0;

        int64_t base_nc = ((int64_t)n * C + c) * (int64_t)H * (int64_t)W;

        for (int ih = h0; ih < h1; ++ih) {
            int64_t row = base_nc + (int64_t)ih * W;
            int iw = w0;

            // Unroll by 4, use read-only cache loads
            for (; iw + 3 < w1; iw += 4) {
                float v0 = ldg_f32(x + row + iw + 0);
                float v1 = ldg_f32(x + row + iw + 1);
                float v2 = ldg_f32(x + row + iw + 2);
                float v3 = ldg_f32(x + row + iw + 3);
                sum += fused_act(v0, sub1, sub2);
                sum += fused_act(v1, sub1, sub2);
                sum += fused_act(v2, sub1, sub2);
                sum += fused_act(v3, sub1, sub2);
                count += 4;
            }
            for (; iw < w1; ++iw) {
                float v = ldg_f32(x + row + iw);
                sum += fused_act(v, sub1, sub2);
                count += 1;
            }
        }

        y[idx] = (count > 0) ? (sum / (float)count) : 0.0f;
    }
}

torch::Tensor fused_tanh_sub_avgpool2d_forward_cuda(
    torch::Tensor x,
    double sub1,
    double sub2,
    int64_t kH, int64_t kW,
    c10::optional<int64_t> sH_opt, c10::optional<int64_t> sW_opt,
    int64_t pH, int64_t pW
) {
    TORCH_CHECK(x.is_cuda(), "fused_tanh_sub_avgpool2d_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "fused_tanh_sub_avgpool2d_forward_cuda: only float32 supported");
    TORCH_CHECK(x.dim() == 4, "fused_tanh_sub_avgpool2d_forward_cuda: expected NCHW 4D input");
    TORCH_CHECK(x.is_contiguous(), "fused_tanh_sub_avgpool2d_forward_cuda: only contiguous NCHW supported");

    const int64_t N64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);

    int64_t sH = sH_opt.has_value() ? sH_opt.value() : kH;
    int64_t sW = sW_opt.has_value() ? sW_opt.value() : kW;

    TORCH_CHECK(kH > 0 && kW > 0, "kernel sizes must be > 0");
    TORCH_CHECK(sH > 0 && sW > 0, "strides must be > 0");
    TORCH_CHECK(pH >= 0 && pW >= 0, "paddings must be >= 0");

    // ceil_mode=False output size
    const int64_t outH64 = (H64 + 2 * pH - kH) / sH + 1;
    const int64_t outW64 = (W64 + 2 * pW - kW) / sW + 1;
    TORCH_CHECK(outH64 >= 0 && outW64 >= 0, "computed output size is negative");

    auto y = torch::empty({N64, C64, outH64, outW64}, x.options());

    constexpr int THREADS = 256;

    int device = x.get_device();
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    // Choose blocks for occupancy; for warp-mapped kernel we want enough blocks to cover lines.
    int blocks = sm_count > 0 ? (sm_count * 8) : 160;

    // Bound by work to avoid too many blocks on small tensors.
    int64_t total = N64 * C64 * outH64 * outW64;
    int max_blocks_by_work = (int)div_up_int64(total, (int64_t)THREADS);
    if (blocks > max_blocks_by_work) blocks = max_blocks_by_work;
    if (blocks < 1) blocks = 1;

    float sub1f = (float)sub1;
    float sub2f = (float)sub2;

    if (kH == 2 && kW == 2 && sH == 2 && sW == 2 && pH == 0 && pW == 0) {
        // Prefer vectorized warp kernel; it naturally handles tail ow.
        // VEC=4 usually good balance of ILP vs regs.
        fused_tanh_sub_avgpool2d_k2s2p0_warpvec_forward<THREADS, 4><<<blocks, THREADS>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N64, (int)C64, (int)H64, (int)W64,
            (int)outH64, (int)outW64,
            sub1f, sub2f
        );
        return y;
    }

    fused_tanh_sub_avgpool2d_generic_forward_kernel_1d<THREADS><<<blocks, THREADS>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N64, (int)C64, (int)H64, (int)W64,
        (int)outH64, (int)outW64,
        (int)kH, (int)kW,
        (int)sH, (int)sW,
        (int)pH, (int)pW,
        sub1f, sub2f
    );
    return y;
}
"""

fused_cpp_src = r"""
#include <torch/extension.h>

torch::Tensor fused_tanh_sub_avgpool2d_forward_cuda(
    torch::Tensor x,
    double sub1,
    double sub2,
    int64_t kH, int64_t kW,
    c10::optional<int64_t> sH_opt, c10::optional<int64_t> sW_opt,
    int64_t pH, int64_t pW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_sub_tanh_sub_avgpool_v5",
    cpp_sources=fused_cpp_src,
    cuda_sources=fused_cuda_src,
    functions=["fused_tanh_sub_avgpool2d_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Keeps Conv2d as-is (cuDNN), fuses:
      x = x - subtract1_value
      x = tanh(x)
      x = x - subtract2_value
      x = avgpool(x)
    into a single CUDA kernel.

    Optimized for AvgPool2d(k=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False).
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)

        self.kH = int(kernel_size_pool)
        self.kW = int(kernel_size_pool)
        self.sH = None
        self.sW = None
        self.pH = 0
        self.pW = 0

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4) or (not x.is_contiguous()):
            y = x - self.subtract1_value
            y = torch.tanh(y)
            y = y - self.subtract2_value
            sH = self.kH if self.sH is None else self.sH
            sW = self.kW if self.sW is None else self.sW
            return F.avg_pool2d(
                y,
                kernel_size=(self.kH, self.kW),
                stride=(sH, sW),
                padding=(self.pH, self.pW),
                ceil_mode=False,
                count_include_pad=False,
            )

        return self.custom_ops.fused_tanh_sub_avgpool2d_forward_cuda(
            x,
            float(self.subtract1_value),
            float(self.subtract2_value),
            self.kH,
            self.kW,
            self.sH,
            self.sW,
            self.pH,
            self.pW,
        )