import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# CUDA/C++ extension: fused (scale1 -> avgpool3d(k=2,s=2) -> bias -> scale2)
# vNext2: improve ILP/occupancy and bias caching
#  - fast path: each thread computes TWO float4 vectors (8 outputs) per loop (Wp%8==0)
#  - bias stored in __constant__ memory (C<=4096 supported; here C=16)
#  - tuned block (TX=64, TY=2) to increase resident CTAs under reg pressure
#  - retains vec4-single and scalar fallbacks
# ============================================================

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAGuard.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
  #define LDG(x) __ldg(x)
#else
  #define LDG(x) (*(x))
#endif

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

// Constant memory for bias (float). Size chosen to cover typical channel counts.
#ifndef CONST_BIAS_MAX
#define CONST_BIAS_MAX 4096
#endif
__constant__ float kBiasConst[CONST_BIAS_MAX];

__device__ __forceinline__ float ldg_mul(const float* __restrict__ p, float s1) {
    return LDG(p) * s1;
}

// ------------------------------------------------------------
// Scalar fallback kernel
// ------------------------------------------------------------
__global__ __launch_bounds__(128, 3) void scale_avgpool3d_bias_scale_scalar_kernel(
    const float* __restrict__ x,     // [N,C,D,H,W]
    float* __restrict__ y,           // [N,C,Dp,Hp,Wp]
    const float* __restrict__ bias,  // [C]
    float scale1,
    float scale2,
    int N, int C, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int ow = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int oh = (int)blockIdx.y;
    int ncod = (int)blockIdx.z; // [0 .. N*C*Dp)
    if (ow >= Wp) return;

    int od = ncod % Dp;
    int t  = ncod / Dp;
    int c  = t % C;
    int n  = t / C;
    if (n >= N) return;

    int base_d = od * 2;
    int base_h = oh * 2;
    int base_w = ow * 2;

    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t DHW = (int64_t)D * HW;
    int64_t CDHW = (int64_t)C * DHW;
    const float* x_ptr = x + (int64_t)n * CDHW + (int64_t)c * DHW;

    float sum = 0.0f;

    #pragma unroll
    for (int pd = 0; pd < 2; ++pd) {
        int id = base_d + pd;
        int64_t d_off = (int64_t)id * HW;
        #pragma unroll
        for (int ph = 0; ph < 2; ++ph) {
            int ih = base_h + ph;
            int64_t h_off = d_off + (int64_t)ih * (int64_t)W + (int64_t)base_w;
            const float* p = x_ptr + h_off;
            sum += ldg_mul(p + 0, scale1);
            sum += ldg_mul(p + 1, scale1);
        }
    }

    float pooled = sum * 0.125f;
    float outv = (pooled + LDG(bias + c)) * scale2;

    int64_t out_idx = ((((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Dp + (int64_t)od) * (int64_t)Hp + (int64_t)oh) * (int64_t)Wp + (int64_t)ow;
    y[out_idx] = outv;
}

// ------------------------------------------------------------
// Vec4 single-output kernel (kept as fallback for Wp%4==0)
// ------------------------------------------------------------
template<int TX, int TY, bool USE_CONST_BIAS>
__global__ __launch_bounds__(TX*TY, 3) void scale_avgpool3d_bias_scale_vec4_2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ bias, // used only if !USE_CONST_BIAS
    float scale1,
    float scale2,
    int N, int C, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int vecW = Wp >> 2;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    int ncod = (int)blockIdx.z;
    int od = ncod % Dp;
    int t  = ncod / Dp;
    int c  = t % C;
    int n  = t / C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * TY + ty;
    if (oh >= Hp) return;

    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t DHW = (int64_t)D * HW;
    int64_t CDHW = (int64_t)C * DHW;

    const float* x_ptr = x + (int64_t)n * CDHW + (int64_t)c * DHW;
    float b = USE_CONST_BIAS ? kBiasConst[c] : LDG(bias + c);

    int base_d = od * 2;
    int base_h = oh * 2;

    int64_t out_base = ((((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Dp + (int64_t)od) * (int64_t)Hp + (int64_t)oh) * (int64_t)Wp;

    int v = (int)blockIdx.x * TX + tx;
    int stride = (int)gridDim.x * TX;

    for (; v < vecW; v += stride) {
        int ow0 = v << 2;
        int base_w0 = ow0 << 1;

        float4 sum4; sum4.x = sum4.y = sum4.z = sum4.w = 0.0f;

        #pragma unroll
        for (int pd = 0; pd < 2; ++pd) {
            int id = base_d + pd;
            int64_t d_off = (int64_t)id * HW;
            #pragma unroll
            for (int ph = 0; ph < 2; ++ph) {
                int ih = base_h + ph;
                int64_t h_off = d_off + (int64_t)ih * (int64_t)W + (int64_t)base_w0;
                const float* p = x_ptr + h_off;

                sum4.x += ldg_mul(p + 0, scale1);
                sum4.x += ldg_mul(p + 1, scale1);
                sum4.y += ldg_mul(p + 2, scale1);
                sum4.y += ldg_mul(p + 3, scale1);
                sum4.z += ldg_mul(p + 4, scale1);
                sum4.z += ldg_mul(p + 5, scale1);
                sum4.w += ldg_mul(p + 6, scale1);
                sum4.w += ldg_mul(p + 7, scale1);
            }
        }

        float4 outv;
        outv.x = ((sum4.x * 0.125f) + b) * scale2;
        outv.y = ((sum4.y * 0.125f) + b) * scale2;
        outv.z = ((sum4.z * 0.125f) + b) * scale2;
        outv.w = ((sum4.w * 0.125f) + b) * scale2;

        float4* y4 = reinterpret_cast<float4*>(y + out_base + ((int64_t)v << 2));
        *y4 = outv;
    }
}

// ------------------------------------------------------------
// Vec4x2 kernel: each thread computes TWO float4 outputs (8 pooled outputs).
// Preconditions: Wp % 8 == 0, y 16B aligned.
// ------------------------------------------------------------
template<int TX, int TY, bool USE_CONST_BIAS>
__global__ __launch_bounds__(TX*TY, 3) void scale_avgpool3d_bias_scale_vec4x2_2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ bias, // used only if !USE_CONST_BIAS
    float scale1,
    float scale2,
    int N, int C, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int vecW = Wp >> 2; // float4 count
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    int ncod = (int)blockIdx.z;
    int od = ncod % Dp;
    int t  = ncod / Dp;
    int c  = t % C;
    int n  = t / C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * TY + ty;
    if (oh >= Hp) return;

    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t DHW = (int64_t)D * HW;
    int64_t CDHW = (int64_t)C * DHW;

    const float* x_ptr = x + (int64_t)n * CDHW + (int64_t)c * DHW;
    float b = USE_CONST_BIAS ? kBiasConst[c] : LDG(bias + c);

    int base_d = od * 2;
    int base_h = oh * 2;

    int64_t out_base = ((((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Dp + (int64_t)od) * (int64_t)Hp + (int64_t)oh) * (int64_t)Wp;

    // Each thread covers v and v+TX*gridDim.x in vec space via stride.
    // But within an iteration, we compute v and v+1 (two adjacent float4) => better ILP.
    int v0 = ((int)blockIdx.x * TX + tx) << 1; // step by 2 vecs per thread
    int stride = ((int)gridDim.x * TX) << 1;

    for (int v = v0; v < vecW; v += stride) {
        int vA = v;
        int vB = v + 1;
        if (vB >= vecW) break; // vecW is even when Wp%8==0

        int owA = vA << 2;      // output w base for A
        int owB = vB << 2;      // output w base for B
        int base_wA = owA << 1; // input base w for A
        int base_wB = owB << 1; // input base w for B

        float4 sumA; sumA.x = sumA.y = sumA.z = sumA.w = 0.0f;
        float4 sumB; sumB.x = sumB.y = sumB.z = sumB.w = 0.0f;

        #pragma unroll
        for (int pd = 0; pd < 2; ++pd) {
            int id = base_d + pd;
            int64_t d_off = (int64_t)id * HW;
            #pragma unroll
            for (int ph = 0; ph < 2; ++ph) {
                int ih = base_h + ph;
                int64_t rowA = d_off + (int64_t)ih * (int64_t)W + (int64_t)base_wA;
                int64_t rowB = d_off + (int64_t)ih * (int64_t)W + (int64_t)base_wB;
                const float* pA = x_ptr + rowA;
                const float* pB = x_ptr + rowB;

                // A lanes
                sumA.x += ldg_mul(pA + 0, scale1); sumA.x += ldg_mul(pA + 1, scale1);
                sumA.y += ldg_mul(pA + 2, scale1); sumA.y += ldg_mul(pA + 3, scale1);
                sumA.z += ldg_mul(pA + 4, scale1); sumA.z += ldg_mul(pA + 5, scale1);
                sumA.w += ldg_mul(pA + 6, scale1); sumA.w += ldg_mul(pA + 7, scale1);

                // B lanes
                sumB.x += ldg_mul(pB + 0, scale1); sumB.x += ldg_mul(pB + 1, scale1);
                sumB.y += ldg_mul(pB + 2, scale1); sumB.y += ldg_mul(pB + 3, scale1);
                sumB.z += ldg_mul(pB + 4, scale1); sumB.z += ldg_mul(pB + 5, scale1);
                sumB.w += ldg_mul(pB + 6, scale1); sumB.w += ldg_mul(pB + 7, scale1);
            }
        }

        float4 outA, outB;
        outA.x = ((sumA.x * 0.125f) + b) * scale2;
        outA.y = ((sumA.y * 0.125f) + b) * scale2;
        outA.z = ((sumA.z * 0.125f) + b) * scale2;
        outA.w = ((sumA.w * 0.125f) + b) * scale2;

        outB.x = ((sumB.x * 0.125f) + b) * scale2;
        outB.y = ((sumB.y * 0.125f) + b) * scale2;
        outB.z = ((sumB.z * 0.125f) + b) * scale2;
        outB.w = ((sumB.w * 0.125f) + b) * scale2;

        float4* yA = reinterpret_cast<float4*>(y + out_base + ((int64_t)vA << 2));
        float4* yB = reinterpret_cast<float4*>(y + out_base + ((int64_t)vB << 2));
        *yA = outA;
        *yB = outB;
    }
}

torch::Tensor scale_avgpool3d_bias_scale_cuda(torch::Tensor x,
                                             torch::Tensor bias_c,
                                             double scale1,
                                             double scale2) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias_c.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(bias_c.dtype() == torch::kFloat32, "only float32 bias supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCDHW");
    TORCH_CHECK(bias_c.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(bias_c.dim() == 1, "bias must be 1D [C]");

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int D = (int)x.size(2);
    int H = (int)x.size(3);
    int W = (int)x.size(4);
    TORCH_CHECK((int)bias_c.numel() == C, "bias must have C elements");

    int Dp = (D - 2) / 2 + 1;
    int Hp = (H - 2) / 2 + 1;
    int Wp = (W - 2) / 2 + 1;
    TORCH_CHECK(Dp > 0 && Hp > 0 && Wp > 0, "pooled output shape must be positive (input too small?)");

    auto y = torch::empty({N, C, Dp, Hp, Wp}, x.options());

    int64_t grid_z64 = (int64_t)N * (int64_t)C * (int64_t)Dp;
    TORCH_CHECK(grid_z64 <= (int64_t)INT_MAX, "grid.z too large");

    bool y_aligned = (((uintptr_t)y.data_ptr<float>() & 0xF) == 0);
    bool use_const_bias = (C <= CONST_BIAS_MAX);

    // Update constant bias (tiny copy; C is small here).
    if (use_const_bias) {
        cudaMemcpyToSymbolAsync(kBiasConst, bias_c.data_ptr<float>(), (size_t)C * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);
    }

    // Prefer vec4x2 when Wp % 8 == 0
    bool vec4x2_ok = ((Wp & 7) == 0) && y_aligned;
    bool vec4_ok   = ((Wp & 3) == 0) && y_aligned;

    if (vec4x2_ok) {
        // TX=64 improves occupancy under reg pressure; TY=2 gives more work per CTA.
        constexpr int TX = 64;
        constexpr int TY = 2;
        dim3 block(TX, TY, 1);

        int vecW = Wp >> 2;
        // each thread handles 2 vectors, so effective threads over vecW is TX*2 per CTA
        int blocks_x = (vecW + (TX * 2 - 1)) / (TX * 2);
        if (blocks_x > 65535) blocks_x = 65535;
        int blocks_y = (Hp + TY - 1) / TY;

        dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, (unsigned)grid_z64);

        if (use_const_bias) {
            scale_avgpool3d_bias_scale_vec4x2_2d_kernel<TX, TY, true><<<grid, block, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float*)nullptr,
                (float)scale1,
                (float)scale2,
                N, C, D, H, W,
                Dp, Hp, Wp
            );
        } else {
            scale_avgpool3d_bias_scale_vec4x2_2d_kernel<TX, TY, false><<<grid, block, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float*)bias_c.data_ptr<float>(),
                (float)scale1,
                (float)scale2,
                N, C, D, H, W,
                Dp, Hp, Wp
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }

    if (vec4_ok) {
        // Vec4 single
        constexpr int TX = 64;
        constexpr int TY = 2;
        dim3 block(TX, TY, 1);

        int vecW = Wp >> 2;
        int blocks_x = (vecW + TX - 1) / TX;
        if (blocks_x > 65535) blocks_x = 65535;
        int blocks_y = (Hp + TY - 1) / TY;

        dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, (unsigned)grid_z64);

        if (use_const_bias) {
            scale_avgpool3d_bias_scale_vec4_2d_kernel<TX, TY, true><<<grid, block, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float*)nullptr,
                (float)scale1,
                (float)scale2,
                N, C, D, H, W,
                Dp, Hp, Wp
            );
        } else {
            scale_avgpool3d_bias_scale_vec4_2d_kernel<TX, TY, false><<<grid, block, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float*)bias_c.data_ptr<float>(),
                (float)scale1,
                (float)scale2,
                N, C, D, H, W,
                Dp, Hp, Wp
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }

    // Scalar fallback
    {
        const int threads = 128;
        int blocks_x = (Wp + threads - 1) / threads;
        dim3 grid((unsigned)blocks_x, (unsigned)Hp, (unsigned)grid_z64);
        scale_avgpool3d_bias_scale_scalar_kernel<<<grid, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)bias_c.data_ptr<float>(),
            (float)scale1,
            (float)scale2,
            N, C, D, H, W,
            Dp, Hp, Wp
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor scale_avgpool3d_bias_scale_cuda(torch::Tensor x,
                                             torch::Tensor bias_c,
                                             double scale1,
                                             double scale2);
"""

custom_ops_lib = load_inline(
    name="custom_conv_transpose3d_scale_avgpool_bias_scale_ops_vNext2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["scale_avgpool3d_bias_scale_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keep ConvTranspose3d (preserves exact semantics incl. conv bias), replace:
      x = x * scale1
      x = AvgPool3d(k=2,s=2)
      x = x + bias
      x = x * scale2
    with one fused CUDA op over NCDHW float32 contiguous tensors.
    Includes CPU fallback to eager ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.scale1 = nn.Parameter(torch.tensor(float(scale1), dtype=torch.float32))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(float(scale2), dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        bias_c = self.bias
        if bias_c.dtype != torch.float32:
            bias_c = bias_c.float()
        bias_c = bias_c.contiguous().view(-1)

        # CPU fallback
        if not x.is_cuda:
            y = x * self.scale1
            y = self.avg_pool(y)
            y = y + bias_c.view(1, -1, 1, 1, 1)
            y = y * self.scale2
            return y

        if (not bias_c.is_cuda) or (bias_c.device != x.device):
            bias_c = bias_c.to(device=x.device)

        # Pass scales as Python floats (kernel args)
        s1 = float(self.scale1.detach().float().cpu().item())
        s2 = float(self.scale2.detach().float().cpu().item())

        return self.custom_ops_lib.scale_avgpool3d_bias_scale_cuda(x, bias_c, s1, s2)