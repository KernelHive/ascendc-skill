import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

static constexpr int FIX_CIN  = 32;
static constexpr int FIX_COUT = 64;
static constexpr int FIX_KH   = 5;
static constexpr int FIX_KW   = 9;
static constexpr int FIX_PH   = 2;
static constexpr int FIX_PW   = 4;
static constexpr int FIX_DH   = 2;
static constexpr int FIX_DW   = 3;
static constexpr int FIX_SH   = 1;
static constexpr int FIX_SW   = 1;

static constexpr int K5 = FIX_KH;
static constexpr int K9 = FIX_KW;
static constexpr int K59 = K5 * K9;

// Optional constant-memory bias for fixed fast-path (64 floats).
__device__ __constant__ float c_bias64[FIX_COUT];

static inline void upload_bias64_to_constant(const float* b_dev, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(c_bias64, b_dev, FIX_COUT * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);
}

// ---------------- Generic fallback kernel ----------------
__global__ void conv2d_forward_generic_kernel(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin,kH,kW]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N,Cout,outH,outW]
    int N, int Cin, int H, int W,
    int Cout, int kH, int kW,
    int outH, int outW,
    int sH, int sW,
    int pH, int pW,
    int dH, int dW,
    bool has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * outH * outW;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % outW; tmp /= outW;
    int oh = tmp % outH; tmp /= outH;
    int oc = tmp % Cout; tmp /= Cout;
    int n  = tmp;

    float acc = has_bias ? ldg_f32(b + oc) : 0.0f;

    int ih0 = oh * sH - pH;
    int iw0 = ow * sW - pW;

    const float* x_n = x + (n * Cin) * H * W;

    for (int ic = 0; ic < Cin; ++ic) {
        const float* x_ic = x_n + ic * H * W;
        const float* w_oc_ic = w + ((oc * Cin + ic) * kH * kW);

        for (int kh = 0; kh < kH; ++kh) {
            int ih = ih0 + kh * dH;
            if ((unsigned)ih >= (unsigned)H) continue;
            const float* x_row = x_ic + ih * W;

            for (int kw = 0; kw < kW; ++kw) {
                int iw = iw0 + kw * dW;
                if ((unsigned)iw >= (unsigned)W) continue;
                float xv = ldg_f32(x_row + iw);
                float wv = ldg_f32(w_oc_ic + kh * kW + kw);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[idx] = acc;
}

// ---------------- Fixed fast-path: OCx4 weights in shared, interior/border split ----------------
__global__ __launch_bounds__(128, 3)
void conv2d_fixed_cin32_k5x9_s1_d2x3_p2x4_oc4_interior_smemw(
    const float* __restrict__ x,   // [N,32,H,W]
    const float* __restrict__ w,   // [64,32,5,9]
    const float* __restrict__ b,   // [64] or nullptr (global)
    float* __restrict__ y,         // [N,64,outH,outW]
    int N, int H, int W,
    int outH, int outW,
    bool has_bias,
    bool use_cbias,
    int oh0, int oh1,              // interior oh range [oh0, oh1)
    int ow0, int ow1               // interior ow range [ow0, ow1)
) {
    int ocv = (int)blockIdx.y; // oc vector (4)
    int n   = (int)blockIdx.z;
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= FIX_COUT) return;

    bool has1 = (oc0 + 1) < FIX_COUT;
    bool has2 = (oc0 + 2) < FIX_COUT;
    bool has3 = (oc0 + 3) < FIX_COUT;

    extern __shared__ float4 sw4[]; // size: 32*K59 float4

    int tid = (int)threadIdx.x;
    int total_w4 = FIX_CIN * K59;

    // Stage weights: sw4[ic*K59 + t] = (w[oc0..oc0+3, ic, t])
    for (int i = tid; i < total_w4; i += (int)blockDim.x) {
        int ic = i / K59;
        int t  = i - ic * K59;

        const float* wbase0 = w + ((oc0 + 0) * FIX_CIN + ic) * K59 + t;
        float a = ldg_f32(wbase0);
        float b1 = has1 ? ldg_f32(w + ((oc0 + 1) * FIX_CIN + ic) * K59 + t) : 0.f;
        float c2 = has2 ? ldg_f32(w + ((oc0 + 2) * FIX_CIN + ic) * K59 + t) : 0.f;
        float d3 = has3 ? ldg_f32(w + ((oc0 + 3) * FIX_CIN + ic) * K59 + t) : 0.f;
        sw4[i] = make_float4(a, b1, c2, d3);
    }
    __syncthreads();

    int intH = oh1 - oh0;
    int intW = ow1 - ow0;
    int interior_pix = intH * intW;

    int base = (int)blockIdx.x * (int)blockDim.x + tid;
    int step = (int)gridDim.x * (int)blockDim.x;

    int64_t x_n_base = (int64_t)n * (int64_t)FIX_CIN * (int64_t)H * (int64_t)W;
    int64_t y_n_base = (int64_t)n * (int64_t)FIX_COUT * (int64_t)outH * (int64_t)outW;

    float bias0 = 0.f, bias1 = 0.f, bias2 = 0.f, bias3 = 0.f;
    if (has_bias) {
        if (use_cbias) {
            bias0 = c_bias64[oc0 + 0];
            bias1 = c_bias64[oc0 + 1];
            bias2 = c_bias64[oc0 + 2];
            bias3 = c_bias64[oc0 + 3];
        } else {
            bias0 = ldg_f32(b + (oc0 + 0));
            bias1 = ldg_f32(b + (oc0 + 1));
            bias2 = ldg_f32(b + (oc0 + 2));
            bias3 = ldg_f32(b + (oc0 + 3));
        }
    }

    for (int p = base; p < interior_pix; p += step) {
        int local_ow = p % intW;
        int local_oh = p / intW;

        int oh = oh0 + local_oh;
        int ow = ow0 + local_ow;

        // Fixed params: stride=1
        int ih0 = oh - FIX_PH;
        int iw0 = ow - FIX_PW;

        float acc0 = bias0, acc1 = bias1, acc2 = bias2, acc3 = bias3;

        #pragma unroll
        for (int ic = 0; ic < FIX_CIN; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * (int64_t)H * (int64_t)W;
            const float4* w4 = sw4 + ic * K59;

            // unrolled 5x9 with dilation (2,3), no bounds checks in interior
            #pragma unroll
            for (int kh = 0; kh < K5; ++kh) {
                int ih = ih0 + kh * FIX_DH;
                int64_t row = x_c_base + (int64_t)ih * (int64_t)W;
                int wt_base = kh * K9;

                #pragma unroll
                for (int kw = 0; kw < K9; ++kw) {
                    int iw = iw0 + kw * FIX_DW;
                    float xv = ldg_f32(x + row + iw);
                    float4 vv = w4[wt_base + kw];
                    acc0 = fmaf(xv, vv.x, acc0);
                    acc1 = fmaf(xv, vv.y, acc1);
                    acc2 = fmaf(xv, vv.z, acc2);
                    acc3 = fmaf(xv, vv.w, acc3);
                }
            }
        }

        int64_t out_pix = (int64_t)oh * (int64_t)outW + (int64_t)ow;
        int64_t out_base0 = y_n_base + (int64_t)(oc0 + 0) * (int64_t)outH * (int64_t)outW + out_pix;
        y[out_base0] = acc0;
        if (has1) y[out_base0 + (int64_t)outH * outW] = acc1;
        if (has2) y[out_base0 + 2LL * (int64_t)outH * outW] = acc2;
        if (has3) y[out_base0 + 3LL * (int64_t)outH * outW] = acc3;
    }
}

__global__ __launch_bounds__(128, 3)
void conv2d_fixed_cin32_k5x9_s1_d2x3_p2x4_oc4_border_smemw(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int H, int W,
    int outH, int outW,
    bool has_bias,
    bool use_cbias,
    int oh_begin, int oh_end,   // [oh_begin, oh_end)
    int ow_begin, int ow_end    // [ow_begin, ow_end)
) {
    int ocv = (int)blockIdx.y;
    int n   = (int)blockIdx.z;
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= FIX_COUT) return;

    bool has1 = (oc0 + 1) < FIX_COUT;
    bool has2 = (oc0 + 2) < FIX_COUT;
    bool has3 = (oc0 + 3) < FIX_COUT;

    extern __shared__ float4 sw4[];
    int tid = (int)threadIdx.x;
    int total_w4 = FIX_CIN * K59;

    for (int i = tid; i < total_w4; i += (int)blockDim.x) {
        int ic = i / K59;
        int t  = i - ic * K59;

        float a = ldg_f32(w + ((oc0 + 0) * FIX_CIN + ic) * K59 + t);
        float b1 = has1 ? ldg_f32(w + ((oc0 + 1) * FIX_CIN + ic) * K59 + t) : 0.f;
        float c2 = has2 ? ldg_f32(w + ((oc0 + 2) * FIX_CIN + ic) * K59 + t) : 0.f;
        float d3 = has3 ? ldg_f32(w + ((oc0 + 3) * FIX_CIN + ic) * K59 + t) : 0.f;
        sw4[i] = make_float4(a, b1, c2, d3);
    }
    __syncthreads();

    int rectH = oh_end - oh_begin;
    int rectW = ow_end - ow_begin;
    int rect_pix = rectH * rectW;

    int base = (int)blockIdx.x * (int)blockDim.x + tid;
    int step = (int)gridDim.x * (int)blockDim.x;

    int64_t x_n_base = (int64_t)n * (int64_t)FIX_CIN * (int64_t)H * (int64_t)W;
    int64_t y_n_base = (int64_t)n * (int64_t)FIX_COUT * (int64_t)outH * (int64_t)outW;

    float bias0 = 0.f, bias1 = 0.f, bias2 = 0.f, bias3 = 0.f;
    if (has_bias) {
        if (use_cbias) {
            bias0 = c_bias64[oc0 + 0];
            bias1 = c_bias64[oc0 + 1];
            bias2 = c_bias64[oc0 + 2];
            bias3 = c_bias64[oc0 + 3];
        } else {
            bias0 = ldg_f32(b + (oc0 + 0));
            bias1 = ldg_f32(b + (oc0 + 1));
            bias2 = ldg_f32(b + (oc0 + 2));
            bias3 = ldg_f32(b + (oc0 + 3));
        }
    }

    for (int p = base; p < rect_pix; p += step) {
        int local_ow = p % rectW;
        int local_oh = p / rectW;

        int oh = oh_begin + local_oh;
        int ow = ow_begin + local_ow;

        if ((unsigned)oh >= (unsigned)outH || (unsigned)ow >= (unsigned)outW) continue;

        int ih0 = oh - FIX_PH;
        int iw0 = ow - FIX_PW;

        float acc0 = bias0, acc1 = bias1, acc2 = bias2, acc3 = bias3;

        #pragma unroll
        for (int ic = 0; ic < FIX_CIN; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * (int64_t)H * (int64_t)W;
            const float4* w4 = sw4 + ic * K59;

            #pragma unroll
            for (int kh = 0; kh < K5; ++kh) {
                int ih = ih0 + kh * FIX_DH;
                if ((unsigned)ih >= (unsigned)H) continue;
                int64_t row = x_c_base + (int64_t)ih * (int64_t)W;
                int wt_base = kh * K9;

                #pragma unroll
                for (int kw = 0; kw < K9; ++kw) {
                    int iw = iw0 + kw * FIX_DW;
                    if ((unsigned)iw >= (unsigned)W) continue;
                    float xv = ldg_f32(x + row + iw);
                    float4 vv = w4[wt_base + kw];
                    acc0 = fmaf(xv, vv.x, acc0);
                    acc1 = fmaf(xv, vv.y, acc1);
                    acc2 = fmaf(xv, vv.z, acc2);
                    acc3 = fmaf(xv, vv.w, acc3);
                }
            }
        }

        int64_t out_pix = (int64_t)oh * (int64_t)outW + (int64_t)ow;
        int64_t out_base0 = y_n_base + (int64_t)(oc0 + 0) * (int64_t)outH * (int64_t)outW + out_pix;
        y[out_base0] = acc0;
        if (has1) y[out_base0 + (int64_t)outH * outW] = acc1;
        if (has2) y[out_base0 + 2LL * (int64_t)outH * outW] = acc2;
        if (has3) y[out_base0 + 3LL * (int64_t)outH * outW] = acc3;
    }
}

torch::Tensor conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW (4D)");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t H   = x.size(2);
    const int64_t W   = x.size(3);

    const int64_t Cout = w.size(0);
    TORCH_CHECK(w.size(1) == Cin, "w.size(1) must equal in_channels");
    const int64_t kH = w.size(2);
    const int64_t kW = w.size(3);

    bool has_bias = (b_opt.defined() && b_opt.numel() > 0);
    const float* b_ptr = nullptr;
    if (has_bias) {
        CHECK_INPUT(b_opt);
        TORCH_CHECK(b_opt.dim() == 1, "bias must be 1D");
        TORCH_CHECK(b_opt.numel() == Cout, "bias numel must equal out_channels");
        b_ptr = b_opt.data_ptr<float>();
    }

    const int64_t outH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const int64_t outW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "computed output size is non-positive");

    auto y = torch::empty({N, Cout, outH, outW}, x.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bool fixed =
        (Cin == FIX_CIN) && (Cout == FIX_COUT) &&
        (kH == FIX_KH) && (kW == FIX_KW) &&
        (sH == FIX_SH) && (sW == FIX_SW) &&
        (pH == FIX_PH) && (pW == FIX_PW) &&
        (dH == FIX_DH) && (dW == FIX_DW);

    if (fixed) {
        bool use_cbias = false;
        if (has_bias) {
            upload_bias64_to_constant(b_ptr, stream);
            use_cbias = true;
        }

        // Interior region where full dilated footprint is in-bounds:
        // need: 0 <= oh - pH and oh - pH + dH*(kH-1) <= H-1
        //   => pH <= oh <= H - 1 - dH*(kH-1) + pH = H - 1 - 8 + 2 = H - 7
        // Similarly width: pW <= ow <= W - 1 - dW*(kW-1) + pW = W - 1 - 24 + 4 = W - 21
        int oh0 = FIX_PH;
        int oh1 = (int)outH;
        int ow0 = FIX_PW;
        int ow1 = (int)outW;

        int oh_max = (int)H - 7;   // inclusive maximum oh for interior
        int ow_max = (int)W - 21;  // inclusive maximum ow for interior

        if (oh1 > (oh_max + 1)) oh1 = oh_max + 1;
        if (ow1 > (ow_max + 1)) ow1 = ow_max + 1;
        if (oh0 < 0) oh0 = 0;
        if (ow0 < 0) ow0 = 0;
        if (oh1 < oh0) oh1 = oh0;
        if (ow1 < ow0) ow1 = ow0;

        int interiorH = oh1 - oh0;
        int interiorW = ow1 - ow0;

        int oc_vecs = (FIX_COUT + 3) / 4; // 16

        // shared weights bytes: 32*45 float4 = 5760 bytes? (actually 1440 float4 = 23040 bytes)
        size_t smem_bytes = (size_t)FIX_CIN * (size_t)K59 * sizeof(float4);

        const int threads = 128;

        auto launch_border = [&](int ob, int oe, int wb, int we) {
            int rh = oe - ob;
            int rw = we - wb;
            if (rh <= 0 || rw <= 0) return;
            int rect_pix = rh * rw;
            int bx = div_up_int(rect_pix, threads);
            if (bx > 4096) bx = 4096;
            if (bx < 1) bx = 1;
            dim3 grid((unsigned)bx, (unsigned)oc_vecs, (unsigned)N);
            conv2d_fixed_cin32_k5x9_s1_d2x3_p2x4_oc4_border_smemw<<<grid, threads, smem_bytes, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                (int)N, (int)H, (int)W,
                (int)outH, (int)outW,
                has_bias,
                use_cbias,
                ob, oe, wb, we
            );
        };

        if (interiorH > 0 && interiorW > 0) {
            int interior_pix = interiorH * interiorW;
            int bx = div_up_int(interior_pix, threads);
            if (bx > 4096) bx = 4096;
            if (bx < 1) bx = 1;

            dim3 grid_in((unsigned)bx, (unsigned)oc_vecs, (unsigned)N);
            conv2d_fixed_cin32_k5x9_s1_d2x3_p2x4_oc4_interior_smemw<<<grid_in, threads, smem_bytes, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                (int)N, (int)H, (int)W,
                (int)outH, (int)outW,
                has_bias,
                use_cbias,
                oh0, oh1, ow0, ow1
            );

            // Top, bottom, left, right borders (avoid overlap with interior)
            launch_border(0, oh0, 0, (int)outW);
            launch_border(oh1, (int)outH, 0, (int)outW);
            launch_border(oh0, oh1, 0, ow0);
            launch_border(oh0, oh1, ow1, (int)outW);
        } else {
            // No interior: everything is border
            int rect_pix = (int)outH * (int)outW;
            int bx = div_up_int(rect_pix, threads);
            if (bx > 4096) bx = 4096;
            if (bx < 1) bx = 1;
            dim3 grid((unsigned)bx, (unsigned)oc_vecs, (unsigned)N);
            conv2d_fixed_cin32_k5x9_s1_d2x3_p2x4_oc4_border_smemw<<<grid, threads, smem_bytes, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                (int)N, (int)H, (int)W,
                (int)outH, (int)outW,
                has_bias,
                use_cbias,
                0, (int)outH, 0, (int)outW
            );
        }
    } else {
        int total = (int)(N * Cout * outH * outW);
        const int threads = 256;
        int blocks = (total + threads - 1) / threads;
        conv2d_forward_generic_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b_ptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W,
            (int)Cout, (int)kH, (int)kW,
            (int)outH, (int)outW,
            (int)sH, (int)sW,
            (int)pH, (int)pW,
            (int)dH, (int)dW,
            has_bias
        );
    }

    return y;
}
"""

cpp_source = r"""
torch::Tensor conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_asym_dil_pad_v8_oc4_smemw_interior_border",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Replacement for nn.Conv2d using an optimized custom CUDA kernel (forward-only).

    Fast-path specialized for:
      Cin=32, Cout=64, kH=5, kW=9, stride=1, padding=(2,4), dilation=(2,3)

    Fallback: generic CUDA kernel otherwise.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super().__init__()

        kH, kW = int(kernel_size[0]), int(kernel_size[1])
        self.sH = int(stride) if not isinstance(stride, (tuple, list)) else int(stride[0])
        self.sW = int(stride) if not isinstance(stride, (tuple, list)) else int(stride[1])
        self.pH = int(padding[0]) if isinstance(padding, (tuple, list)) else int(padding)
        self.pW = int(padding[1]) if isinstance(padding, (tuple, list)) else int(padding)
        self.dH = int(dilation[0]) if isinstance(dilation, (tuple, list)) else int(dilation[0])
        self.dW = int(dilation[1]) if isinstance(dilation, (tuple, list)) else int(dilation[1])

        w = torch.empty((out_channels, in_channels, kH, kW), dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.empty((out_channels,), dtype=torch.float32)
            fan_in = in_channels * kH * kW
            bound = 1.0 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter("bias", None)

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        if not w.is_cuda:
            w = w.to(device=x.device)
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        if self.bias is None:
            b_opt = torch.empty((0,), device=x.device, dtype=torch.float32)
        else:
            b = self.bias
            if not b.is_cuda:
                b = b.to(device=x.device)
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()
            b_opt = b

        return self.custom_ops.conv2d_forward_cuda(
            x, w, b_opt,
            self.sH, self.sW,
            self.pH, self.pW,
            self.dH, self.dW,
        )