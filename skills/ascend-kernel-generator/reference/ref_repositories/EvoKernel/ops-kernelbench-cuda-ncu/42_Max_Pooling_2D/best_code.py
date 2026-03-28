import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __device__ __forceinline__ float neg_inf() { return -INFINITY; }

static __device__ __forceinline__ float ld_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Generic fallback (any params): 1 thread = 1 output
__global__ void maxpool2d_forward_generic_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    int dH, int dW
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int t  = idx / outW;
    int oh = t % outH;
    t /= outH;
    int c  = t % C;
    int n  = t / C;

    int hstart = oh * sH - pH;
    int wstart = ow * sW - pW;

    float m = neg_inf();

    for (int kh = 0; kh < kH; ++kh) {
        int ih = hstart + kh * dH;
        if ((unsigned)ih >= (unsigned)H) continue;
        const float* row = x + ((n * C + c) * H + ih) * W;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = wstart + kw * dW;
            if ((unsigned)iw >= (unsigned)W) continue;
            float v = ld_ro(row + iw);
            m = v > m ? v : m;
        }
    }

    y[((n * C + c) * outH + oh) * outW + ow] = m;
}

// Fast path specialized for k=4,s=1,d=1. No SMEM. 1 thread computes 2 outputs (ow, ow+1).
// Block: (TILE_OW/2 * TILE_OH) threads; grid.z = N*C.
template<int TILE_OH, int TILE_OW>
__global__ __launch_bounds__(256, 2) void maxpool2d_forward_k4s1d1_2ow_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int C, int H, int W,
    int outH, int outW,
    int pH, int pW
) {
    int nc = (int)blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;

    int oh0 = (int)blockIdx.y * TILE_OH;
    int ow0 = (int)blockIdx.x * TILE_OW;

    // thread mapping: threads form (TILE_OH, TILE_OW/2)
    constexpr int TW2 = TILE_OW / 2;
    int tid = (int)threadIdx.x;
    int ty = tid / TW2;      // 0..TILE_OH-1
    int tx2 = tid - ty * TW2; // 0..TW2-1

    if (ty >= TILE_OH) return;

    int oh = oh0 + ty;
    int ow = ow0 + (tx2 * 2);
    if ((unsigned)oh >= (unsigned)outH || (unsigned)ow >= (unsigned)outW) return;

    // Input origin for ow
    int ih0 = oh - pH;
    int iw0 = ow - pW;

    const float* xbase = x + ((n * C + c) * H) * W;
    float* ybase = y + ((n * C + c) * outH + oh) * outW;

    // Helper to load with padding
    auto load_pad = [&](int ih, int iw) -> float {
        if (((unsigned)ih < (unsigned)H) && ((unsigned)iw < (unsigned)W)) {
            return ld_ro(xbase + ih * W + iw);
        }
        return neg_inf();
    };

    // Interior fast checks (avoid per-element bounds): need ih0..ih0+3, iw0..iw0+4 (because ow+1 uses iw0+1..iw0+4)
    bool interior_h = (unsigned)ih0 < (unsigned)(H - 3);
    bool interior_w = (unsigned)iw0 < (unsigned)(W - 4);
    bool interior = interior_h && interior_w;

    float m0 = neg_inf();
    float m1 = neg_inf();

    if (interior) {
        // For each of 4 rows, load 5 contiguous floats: iw0..iw0+4
        // Then compute max for ow (0..3) and ow+1 (1..4)
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float* row = xbase + (ih0 + r) * W + iw0;

            // Use float4 when aligned; always safe to read row[0..3] and scalar row[4]
            float a0, a1, a2, a3, a4;
            if ((((uintptr_t)row & 0xF) == 0)) {
                float4 v4 = *reinterpret_cast<const float4*>(row);
                a0 = v4.x; a1 = v4.y; a2 = v4.z; a3 = v4.w;
                a4 = ld_ro(row + 4);
            } else {
                a0 = ld_ro(row + 0);
                a1 = ld_ro(row + 1);
                a2 = ld_ro(row + 2);
                a3 = ld_ro(row + 3);
                a4 = ld_ro(row + 4);
            }

            // max for ow: a0..a3
            float r0 = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
            // max for ow+1: a1..a4
            float r1 = fmaxf(fmaxf(a1, a2), fmaxf(a3, a4));

            m0 = fmaxf(m0, r0);
            m1 = fmaxf(m1, r1);
        }
    } else {
        // Border-safe scalar path (still exploits overlap)
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int ih = ih0 + r;
            float a0 = load_pad(ih, iw0 + 0);
            float a1 = load_pad(ih, iw0 + 1);
            float a2 = load_pad(ih, iw0 + 2);
            float a3 = load_pad(ih, iw0 + 3);
            float a4 = load_pad(ih, iw0 + 4);

            float r0 = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
            float r1 = fmaxf(fmaxf(a1, a2), fmaxf(a3, a4));

            m0 = fmaxf(m0, r0);
            m1 = fmaxf(m1, r1);
        }
    }

    // Store ow
    ybase[ow] = m0;
    // Store ow+1 if in range
    if ((unsigned)(ow + 1) < (unsigned)outW) ybase[ow + 1] = m1;
}

torch::Tensor max_pool2d_forward_cuda(
    torch::Tensor x,
    int64_t kH, int64_t kW,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    TORCH_CHECK(kH > 0 && kW > 0, "kernel size must be > 0");
    TORCH_CHECK(sH > 0 && sW > 0, "stride must be > 0");
    TORCH_CHECK(dH > 0 && dW > 0, "dilation must be > 0");

    const int64_t outH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const int64_t outW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "computed output size is non-positive");

    auto y = torch::empty({N, C, outH, outW}, x.options());

    const bool fast = (kH == 4 && kW == 4 &&
                       sH == 1 && sW == 1 &&
                       dH == 1 && dW == 1);

    if (fast) {
        // Choose tile to balance launch overhead vs. occupancy
        // TILE_OW must be even (2 outputs/thread along width)
        constexpr int TILE_OH = 8;
        constexpr int TILE_OW = 32; // 32-wide tile, 16 threads per row => 128 threads/block

        dim3 block((TILE_OH * (TILE_OW / 2)), 1, 1); // 8*16 = 128
        dim3 grid(
            (unsigned)((outW + TILE_OW - 1) / TILE_OW),
            (unsigned)((outH + TILE_OH - 1) / TILE_OH),
            (unsigned)(N * C)
        );

        maxpool2d_forward_k4s1d1_2ow_kernel<TILE_OH, TILE_OW><<<grid, block>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)C, (int)H, (int)W,
            (int)outH, (int)outW,
            (int)pH, (int)pW
        );
        return y;
    }

    const int threads = 256;
    const int64_t total = N * C * outH * outW;
    const int blocks = (int)((total + threads - 1) / threads);

    maxpool2d_forward_generic_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)outH, (int)outW,
        (int)kH, (int)kW,
        (int)sH, (int)sW,
        (int)pH, (int)pW,
        (int)dH, (int)dW
    );

    return y;
}
"""

cpp_source = r"""
torch::Tensor max_pool2d_forward_cuda(
    torch::Tensor x,
    int64_t kH, int64_t kW,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_maxpool2d_opt_ilp2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["max_pool2d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Replacement for nn.MaxPool2d using a custom CUDA kernel (forward-only).
    Assumes input is CUDA, contiguous, and float32.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kH = int(kernel_size)
        self.kW = int(kernel_size)
        self.sH = int(stride)
        self.sW = int(stride)
        self.pH = int(padding)
        self.pW = int(padding)
        self.dH = int(dilation)
        self.dW = int(dilation)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops.max_pool2d_forward_cuda(
            x,
            self.kH, self.kW,
            self.sH, self.sW,
            self.pH, self.pW,
            self.dH, self.dW,
        )