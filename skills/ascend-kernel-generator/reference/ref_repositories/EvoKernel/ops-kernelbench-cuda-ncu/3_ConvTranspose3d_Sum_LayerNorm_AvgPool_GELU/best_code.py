import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# Fused tail (v9):
#   x (post ConvTranspose3d): [N,C,D,H,W] contiguous float32
#   y: [N,C,Do,Ho,Wo] where Do=(D-2)/2+1 etc (AvgPool3d k=2 s=2 p=0)
#
# Semantics MUST match PyTorch:
#   x = x + sum_weight (scalar)
#   x = LayerNorm(normalized_shape=(out_channels,)) on NCDHW => normalize over last dim W
#   x = AvgPool3d(k=2,s=2,p=0) over D/H/W
#   x = GELU() default approximate='none' => exact erf formulation
#
# Fast path specialization for W==64 (Wo==32):
#   - Stage gamma/beta into shared (2*64 floats).
#   - Each warp loads its row (64 floats) of (x+sum_weight) into shared once.
#   - Compute LN stats from shared (warp reduce over 64 elements).
#   - Use shared values again for pooled-width contributions.
#   - Cross-warp reduction for pooling uses warp shuffles (no atomics, minimal barriers).
#
# Fallback: baseline kernel (general W), same semantics.
# ============================================================

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")

// Exact GELU (approximate='none'):
// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
__device__ __forceinline__ float gelu_pytorch_exact(float x) {
    const float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Baseline kernel (general W): correct, but slower. Kept as fallback.
__global__ __launch_bounds__(256, 2) void sum_lnW_pool2_gelu_kernel_baseline(
    const float* __restrict__ x,
    const float* __restrict__ sum_w,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int D, int H, int W,
    int Do, int Ho, int Wo,
    float eps
){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)Do * (int64_t)Ho * (int64_t)Wo;
    if (idx >= total) return;

    int64_t t = idx;
    int ow = (int)(t % Wo); t /= Wo;
    int oh = (int)(t % Ho); t /= Ho;
    int od = (int)(t % Do); t /= Do;
    int c  = (int)(t % C);  t /= C;
    int n  = (int)t;

    float sw = __ldg(sum_w);

    int id0 = od * 2;
    int ih0 = oh * 2;
    int iw0 = ow * 2;

    int64_t strideH = (int64_t)W;
    int64_t strideD = (int64_t)H * (int64_t)W;
    int64_t strideC = (int64_t)D * (int64_t)H * (int64_t)W;
    int64_t strideN = (int64_t)C * (int64_t)D * (int64_t)H * (int64_t)W;

    float acc = 0.0f;

    #pragma unroll
    for (int kd = 0; kd < 2; ++kd) {
        int id = id0 + kd;
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            int ih = ih0 + kh;
            #pragma unroll
            for (int kw = 0; kw < 2; ++kw) {
                int iw = iw0 + kw;

                int64_t row0 = (int64_t)n * strideN
                             + (int64_t)c * strideC
                             + (int64_t)id * strideD
                             + (int64_t)ih * strideH;

                float sum = 0.0f;
                float sumsq = 0.0f;

                int ww = 0;
                int w4 = W & ~3;
                for (; ww < w4; ww += 4) {
                    float v0 = x[row0 + (int64_t)(ww + 0)] + sw;
                    float v1 = x[row0 + (int64_t)(ww + 1)] + sw;
                    float v2 = x[row0 + (int64_t)(ww + 2)] + sw;
                    float v3 = x[row0 + (int64_t)(ww + 3)] + sw;
                    sum += (v0 + v1) + (v2 + v3);
                    sumsq = fmaf(v0, v0, sumsq);
                    sumsq = fmaf(v1, v1, sumsq);
                    sumsq = fmaf(v2, v2, sumsq);
                    sumsq = fmaf(v3, v3, sumsq);
                }
                for (; ww < W; ++ww) {
                    float v = x[row0 + (int64_t)ww] + sw;
                    sum += v;
                    sumsq = fmaf(v, v, sumsq);
                }

                float mean = sum * (1.0f / (float)W);
                float var  = sumsq * (1.0f / (float)W) - mean * mean;
                var = var < 0.0f ? 0.0f : var;
                float inv_std = rsqrtf(var + eps);

                float xv = x[row0 + (int64_t)iw] + sw;
                float nrm = (xv - mean) * inv_std;
                float g = __ldg(gamma + iw);
                float b = __ldg(beta + iw);
                float ln_aff = fmaf(nrm, g, b);

                acc += ln_aff;
            }
        }
    }

    float pooled = acc * (1.0f / 8.0f);
    y[idx] = gelu_pytorch_exact(pooled);
}

// W==64 fast path kernel (v9)
__global__ __launch_bounds__(128, 4) void sum_lnW64_pool2_gelu_kernel_v9(
    const float* __restrict__ x,     // [N,C,D,H,64]
    const float* __restrict__ sum_w, // [1]
    const float* __restrict__ gamma, // [64]
    const float* __restrict__ beta,  // [64]
    float* __restrict__ y,           // [N,C,Do,Ho,32]
    int N, int C, int D, int H,
    int Do, int Ho,
    float eps
){
    constexpr int W = 64;
    constexpr int Wo = 32;

    // Shared: gamma/beta and 4 rows of (x+sw)
    __shared__ float sh_g[W];
    __shared__ float sh_b[W];
    __shared__ float sh_x[4][W]; // 4 warps each owns one row

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..3

    // Load gamma/beta once per block (all threads cooperate)
    if (tid < W) {
        sh_g[tid] = __ldg(gamma + tid);
        sh_b[tid] = __ldg(beta + tid);
    }

    // Decode block -> (n,c,od,oh)
    int64_t b = (int64_t)blockIdx.x; // 0 .. N*C*Do*Ho-1
    int64_t per_nc = (int64_t)Do * (int64_t)Ho;
    int64_t nc = b / per_nc;
    int64_t rem = b - nc * per_nc;
    int od = (int)(rem / Ho);
    int oh = (int)(rem - (int64_t)od * Ho);

    int n = (int)(nc / C);
    int c = (int)(nc - (int64_t)n * C);
    if (n >= N) return;

    float sw = __ldg(sum_w);

    int id0 = od * 2;
    int ih0 = oh * 2;

    // warp -> (kd,kh)
    int kd = (warp >> 1); // 0,0,1,1
    int kh = (warp & 1);  // 0,1,0,1
    int id = id0 + kd;
    int ih = ih0 + kh;

    // Strides for contiguous NCDHW with W=64
    int64_t strideH = (int64_t)W;
    int64_t strideD = (int64_t)H * (int64_t)W;
    int64_t strideC = (int64_t)D * (int64_t)H * (int64_t)W;
    int64_t strideN = (int64_t)C * (int64_t)D * (int64_t)H * (int64_t)W;

    int64_t row0 = (int64_t)n * strideN
                 + (int64_t)c * strideC
                 + (int64_t)id * strideD
                 + (int64_t)ih * strideH;

    // Cooperative load x row into shared as (x + sw).
    // Use float4 loads when aligned; cover full 64 values per warp:
    // lanes 0..15 load float4 each => 16 * 4 = 64 floats.
    const float* row_ptr = x + row0;
    uintptr_t addr = (uintptr_t)row_ptr;
    bool aligned16 = ((addr & 0xF) == 0);

    if (lane < 16) {
        if (aligned16) {
            const float4* p4 = reinterpret_cast<const float4*>(row_ptr);
            float4 v4 = __ldg(p4 + lane);
            // write to sh_x[warp][4*lane + i]
            int base = lane * 4;
            sh_x[warp][base + 0] = v4.x + sw;
            sh_x[warp][base + 1] = v4.y + sw;
            sh_x[warp][base + 2] = v4.z + sw;
            sh_x[warp][base + 3] = v4.w + sw;
        } else {
            int base = lane * 4;
            sh_x[warp][base + 0] = __ldg(row_ptr + base + 0) + sw;
            sh_x[warp][base + 1] = __ldg(row_ptr + base + 1) + sw;
            sh_x[warp][base + 2] = __ldg(row_ptr + base + 2) + sw;
            sh_x[warp][base + 3] = __ldg(row_ptr + base + 3) + sw;
        }
    }

    __syncthreads(); // ensure sh_g/sh_b and sh_x are ready

    // LayerNorm stats from shared row across W=64.
    float v0 = sh_x[warp][lane];
    float v1 = sh_x[warp][lane + 32];
    float lsum = v0 + v1;
    float lsq  = fmaf(v0, v0, v1 * v1);

    float sum = warp_reduce_sum(lsum);
    float sumsq = warp_reduce_sum(lsq);
    sum = __shfl_sync(0xffffffff, sum, 0);
    sumsq = __shfl_sync(0xffffffff, sumsq, 0);

    float mean = sum * (1.0f / 64.0f);
    float var  = sumsq * (1.0f / 64.0f) - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    float inv_std = rsqrtf(var + eps);

    // Each lane<32 produces its warp's partial sum for output wo==lane:
    // widths iw=2*wo and iw+1.
    float contrib = 0.0f;
    if (lane < Wo) {
        int iw = lane * 2;
        float a0 = sh_x[warp][iw];
        float a1 = sh_x[warp][iw + 1];

        float n0 = (a0 - mean) * inv_std;
        float n1 = (a1 - mean) * inv_std;

        float g0 = sh_g[iw];
        float b0 = sh_b[iw];
        float g1 = sh_g[iw + 1];
        float b1 = sh_b[iw + 1];

        contrib = (fmaf(n0, g0, b0) + fmaf(n1, g1, b1));
    }

    // Cross-warp reduction using warp0 as aggregator via shuffles:
    // For a fixed lane, collect contrib from warps 0..3.
    // We can do this by broadcasting contrib within each warp to lane0,
    // then warp0 reads those lane0 values via shared memory? But we avoid extra shared
    // by using one more shared array of 4*32 floats? No: we can use one shared float per (warp,lane)
    // but that's what we want to avoid. Instead, use a minimal shared of 4*32 floats is acceptable,
    // but v4 already did it; here we avoid it by using cooperative groups? Not allowed.
    // So we use a tiny shared just for 4*32 floats (same size as v4) is not the goal.
    // Alternative: write each warp's contrib to registers of warp0 is impossible.
    // Therefore, use a very small shared: 4*32 floats, but we already saved big global traffic,
    // and keep only one barrier (we already had one). This is still better than v4 because
    // we eliminated repeated global loads and the shared write is unavoidable to combine warps.
    __shared__ float sh_acc[4][Wo];
    if (lane < Wo) sh_acc[warp][lane] = contrib;
    __syncthreads();

    if (warp == 0 && lane < Wo) {
        float acc = sh_acc[0][lane] + sh_acc[1][lane] + sh_acc[2][lane] + sh_acc[3][lane];
        float pooled = acc * (1.0f / 8.0f);
        float outv = gelu_pytorch_exact(pooled);
        int64_t out_off = ((((int64_t)n * C + c) * (int64_t)Do + od) * (int64_t)Ho + oh) * (int64_t)Wo + lane;
        y[out_off] = outv;
    }
}

torch::Tensor sum_layernormW_avgpool2_gelu_cuda_v9(
    torch::Tensor x,
    torch::Tensor sum_weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
){
    CHECK_CUDA(x); CHECK_CUDA(sum_weight); CHECK_CUDA(gamma); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(sum_weight); CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta);
    CHECK_F32(x); CHECK_F32(sum_weight); CHECK_F32(gamma); CHECK_F32(beta);

    TORCH_CHECK(x.dim() == 5, "x must be [N,C,D,H,W]");
    TORCH_CHECK(sum_weight.numel() == 1, "sum_weight must be scalar/1-element");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(gamma.numel() == x.size(4), "gamma must have W elements (LayerNorm over last dim)");
    TORCH_CHECK(beta.numel() == x.size(4), "beta must have W elements (LayerNorm over last dim)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    TORCH_CHECK(D >= 2 && H >= 2 && W >= 2, "input too small for AvgPool3d(k=2,s=2)");

    int64_t Do = (D - 2) / 2 + 1;
    int64_t Ho = (H - 2) / 2 + 1;
    int64_t Wo = (W - 2) / 2 + 1;
    TORCH_CHECK(Do > 0 && Ho > 0 && Wo > 0, "computed pooled shape invalid");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty({N, C, Do, Ho, Wo}, x.options());

    if (W == 64 && Wo == 32) {
        int64_t blocks = N * C * Do * Ho;
        TORCH_CHECK(blocks <= (int64_t)INT_MAX, "grid too large");
        dim3 grid((unsigned)blocks);
        dim3 block(128);
        sum_lnW64_pool2_gelu_kernel_v9<<<grid, block>>>(
            x.data_ptr<float>(),
            sum_weight.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)C, (int)D, (int)H,
            (int)Do, (int)Ho,
            (float)eps
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }

    // fallback
    int64_t total = y.numel();
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    sum_lnW_pool2_gelu_kernel_baseline<<<blocks, threads>>>(
        x.data_ptr<float>(),
        sum_weight.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)C, (int)D, (int)H, (int)W,
        (int)Do, (int)Ho, (int)Wo,
        (float)eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor sum_layernormW_avgpool2_gelu_cuda_v9(
    torch::Tensor x,
    torch::Tensor sum_weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convt3d_sum_lnW_pool2_gelu_v9",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["sum_layernormW_avgpool2_gelu_cuda_v9"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Custom CUDA replacement for the tail of:
      ConvTranspose3d -> (+sum_weight) -> LayerNorm(norm_shape=(out_channels,))
      -> AvgPool3d(2,2,2) -> GELU

    Semantic note:
      PyTorch LayerNorm(normalized_shape=(out_channels,)) on N,C,D,H,W normalizes over last dim W.
      This fused kernel matches that behavior and requires W == out_channels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        sum_weight,
        norm_shape,
        pool_kernel_size,
    ):
        super().__init__()
        self.custom_ops = custom_ops_lib

        self.conv_transpose = nn.ConvTranspose3d(
            int(in_channels),
            int(out_channels),
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        self.sum_weight = nn.Parameter(torch.tensor(float(sum_weight), dtype=torch.float32))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

        ns = tuple(int(v) for v in (norm_shape if isinstance(norm_shape, (list, tuple)) else (norm_shape,)))
        if len(ns) != 1:
            raise ValueError("Fused op supports LayerNorm with 1D normalized_shape only")
        if ns[0] != int(out_channels):
            raise ValueError("Fused op expects LayerNorm normalized_shape == (out_channels,)")

        pk = pool_kernel_size
        if isinstance(pk, int):
            pk = (pk, pk, pk)
        else:
            pk = tuple(int(v) for v in pk)
        if pk != (2, 2, 2):
            raise ValueError("Fused op supports AvgPool3d kernel_size == (2,2,2) only")

        st = self.avg_pool.stride
        if isinstance(st, int):
            st = (st, st, st)
        else:
            st = tuple(int(v) for v in st)
        if st != (2, 2, 2):
            raise ValueError("Fused op supports AvgPool3d stride == (2,2,2) only")

        pd = self.avg_pool.padding
        if isinstance(pd, int):
            pd = (pd, pd, pd)
        else:
            pd = tuple(int(v) for v in pd)
        if pd != (0, 0, 0):
            raise ValueError("Fused op supports AvgPool3d padding == 0 only")
        if bool(self.avg_pool.ceil_mode):
            raise ValueError("Fused op supports AvgPool3d ceil_mode=False only")

        self.eps = float(self.norm.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if not x.is_cuda:
            raise RuntimeError("ModelNew supports CUDA tensors only")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        dev = x.device
        sw = self.sum_weight.to(device=dev, dtype=torch.float32).contiguous()
        gamma = self.norm.weight.to(device=dev, dtype=torch.float32).contiguous()
        beta = self.norm.bias.to(device=dev, dtype=torch.float32).contiguous()

        if x.size(-1) != gamma.numel():
            raise RuntimeError(
                f"Fused LayerNorm expects last dim W == {gamma.numel()}, but got W={x.size(-1)}"
            )

        return self.custom_ops.sum_layernormW_avgpool2_gelu_cuda_v9(
            x,
            sw,
            gamma,
            beta,
            float(self.eps),
        )