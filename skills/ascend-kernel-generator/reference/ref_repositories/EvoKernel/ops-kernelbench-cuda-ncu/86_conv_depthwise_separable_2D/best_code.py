import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline int64_t conv_out_size(int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg(const float* p) { return *p; }
#endif

// ---------------- Depthwise conv2d (generic fallback) ----------------

__global__ void depthwise_conv2d_forward_generic(
    const float* __restrict__ x,   // [N,C,H,W]
    const float* __restrict__ w,   // [C,1,kH,kW]
    const float* __restrict__ b,   // [C] or nullptr
    float* __restrict__ y,         // [N,C,outH,outW]
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
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % C;
    int n  = idx / (outW * outH * C);

    int in_h0 = oh * sH - pH;
    int in_w0 = ow * sW - pW;

    float acc = (b != nullptr) ? ldg(b + c) : 0.0f;

    const float* x_nc = x + ((n * C + c) * H * W);
    int w_base = c * kH * kW;

    for (int kh = 0; kh < kH; ++kh) {
        int ih = in_h0 + kh * dH;
        if ((unsigned)ih >= (unsigned)H) continue;
        int x_row = ih * W;
        int w_row = w_base + kh * kW;
        #pragma unroll 1
        for (int kw = 0; kw < kW; ++kw) {
            int iw = in_w0 + kw * dW;
            if ((unsigned)iw >= (unsigned)W) continue;
            acc = fmaf(ldg(x_nc + x_row + iw), ldg(w + w_row + kw), acc);
        }
    }
    y[((n * C + c) * outH + oh) * outW + ow] = acc;
}

torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [C,1,kH,kW]");
    TORCH_CHECK(weight.size(1) == 1, "depthwise weight second dim must be 1");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(weight.size(0) == C, "weight.size(0) must equal x.size(1)");
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    int64_t outH = conv_out_size(H, pH, dH, kH, sH);
    int64_t outW = conv_out_size(W, pW, dW, kW, sW);

    const float* bptr = nullptr;
    torch::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must be [C]");
        bptr = bias.data_ptr<float>();
    }

    auto y = torch::empty({N, C, outH, outW}, x.options());

    const int threads = 256;
    const int64_t total = N * C * outH * outW;
    const int blocks = (int)((total + threads - 1) / threads);

    depthwise_conv2d_forward_generic<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)outH, (int)outW,
        (int)kH, (int)kW,
        (int)sH, (int)sW,
        (int)pH, (int)pW,
        (int)dH, (int)dW
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ---------------- Pointwise conv2d (generic fallback) ----------------

__global__ void pointwise_conv2d_forward_kernel(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N,Cout,H,W]
    int N, int Cin, int H, int W,
    int Cout
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * H * W;
    if (idx >= total) return;

    int ow = idx % W;
    int oh = (idx / W) % H;
    int co = (idx / (W * H)) % Cout;
    int n  = idx / (W * H * Cout);

    float acc = (b != nullptr) ? ldg(b + co) : 0.0f;

    int hw = oh * W + ow;
    const float* x_base = x + (n * Cin) * (int64_t)H * (int64_t)W + hw;
    const float* w_base = w + (int64_t)co * Cin;

    #pragma unroll 1
    for (int ci = 0; ci < Cin; ++ci) {
        acc = fmaf(ldg(x_base + (int64_t)ci * H * W), ldg(w_base + ci), acc);
    }
    y[((n * Cout + co) * (int64_t)H + oh) * W + ow] = acc;
}

torch::Tensor pointwise_conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [Cout,Cin,1,1]");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "pointwise weight must be 1x1");

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t Cout = weight.size(0);

    const float* bptr = nullptr;
    torch::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == Cout, "bias must be [Cout]");
        bptr = bias.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, H, W}, x.options());
    const int threads = 256;
    const int64_t total = N * Cout * H * W;
    const int blocks = (int)((total + threads - 1) / threads);

    pointwise_conv2d_forward_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)H, (int)W,
        (int)Cout
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ---------------- Fused DW(3x3 s1 p1 d1) + PW(1x1) correct Cin-reduction ----------------
//
// Block covers a small (oh, ow) tile and one Cout tile.
// Threads iterate over Cin in a strided way and accumulate partial sums for COUT_TILE output channels.
// Then block reduces partial sums across threads (only for its Cout tile and its pixels).
//
// Key choices to reduce register pressure vs baseline:
// - Small COUT_TILE (8) so per-thread accumulators are [8][WPT] (manageable)
// - No shared staging of full COUT_TILE*Cin weights (stream weights)
// - Depthwise computed with register sliding window for WPT pixels
// - Reduction uses shared memory but only for partial sums (not weights)
//
// Note: This is forward-only, float32, NCHW, fixed DW params 3x3 s1 p1 d1.

template<int BX, int BY, int WPT, int COUT_TILE, bool HAS_DW_BIAS, bool HAS_PW_BIAS>
__global__ __launch_bounds__(BX*BY, 3)
void fused_dw3x3_pw1x1_cin_reduce(
    const float* __restrict__ x,        // [N,Cin,H,W]
    const float* __restrict__ dw_w,     // [Cin,1,3,3]
    const float* __restrict__ dw_b,     // [Cin] or nullptr
    const float* __restrict__ pw_w,     // [Cout,Cin] (from [Cout,Cin,1,1])
    const float* __restrict__ pw_b,     // [Cout] or nullptr
    float* __restrict__ y,              // [N,Cout,H,W]
    int N, int Cin, int H, int W,
    int Cout
) {
    const int HW = H * W;

    // z tiles over (n, cout_tile)
    int ct_tiles = (Cout + COUT_TILE - 1) / COUT_TILE;
    int bz = (int)blockIdx.z;
    int n = bz / ct_tiles;
    int ct = bz - n * ct_tiles;
    if (n >= N) return;
    int co_base = ct * COUT_TILE;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int tid = ty * BX + tx;
    int nthreads = BX * BY;

    int oh = (int)blockIdx.y * BY + ty;
    int ow0 = ((int)blockIdx.x * BX + tx) * WPT;

    // Accumulators for this thread: COUT_TILE x WPT
    float acc[COUT_TILE][WPT];
    #pragma unroll
    for (int co = 0; co < COUT_TILE; ++co) {
        float b0 = 0.f;
        if constexpr (HAS_PW_BIAS) {
            int gco = co_base + co;
            if (gco < Cout) b0 = ldg(pw_b + gco);
        }
        #pragma unroll
        for (int i = 0; i < WPT; ++i) acc[co][i] = b0;
    }

    if (oh >= H) {
        // nothing to do (outside), but keep behavior defined
        return;
    }

    const float* x_n = x + (int64_t)n * Cin * (int64_t)HW;
    float* y_n = y + (int64_t)n * Cout * (int64_t)HW;

    // Iterate over Cin in a strided manner per thread to increase MLP and reduce redundant work.
    // Each thread computes DW at its pixels for its subset of Cin, then accumulates into its acc.
    for (int c = tid; c < Cin; c += nthreads) {
        const float* x_nc = x_n + (int64_t)c * HW;

        // load dw weights once
        const float* dwc = dw_w + c * 9;
        float w00 = ldg(dwc + 0), w01 = ldg(dwc + 1), w02 = ldg(dwc + 2);
        float w10 = ldg(dwc + 3), w11 = ldg(dwc + 4), w12 = ldg(dwc + 5);
        float w20 = ldg(dwc + 6), w21 = ldg(dwc + 7), w22 = ldg(dwc + 8);
        float dbv = 0.0f;
        if constexpr (HAS_DW_BIAS) dbv = ldg(dw_b + c);

        int ih0 = oh - 1;

        auto ld_x = [&](int ih, int iw)->float {
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                return ldg(x_nc + ih * W + iw);
            }
            return 0.0f;
        };

        // Sliding window for WPT pixels
        int iw_base = ow0 - 1;

        // prefetch for first output in this thread
        float a0 = ld_x(ih0 + 0, iw_base + 0);
        float a1 = ld_x(ih0 + 0, iw_base + 1);
        float a2 = ld_x(ih0 + 0, iw_base + 2);

        float b0 = ld_x(ih0 + 1, iw_base + 0);
        float b1 = ld_x(ih0 + 1, iw_base + 1);
        float b2 = ld_x(ih0 + 1, iw_base + 2);

        float c0 = ld_x(ih0 + 2, iw_base + 0);
        float c1 = ld_x(ih0 + 2, iw_base + 1);
        float c2 = ld_x(ih0 + 2, iw_base + 2);

        // Load PW weights for this Cin and this Cout tile (8 scalars) once per Cin
        float pwv[COUT_TILE];
        #pragma unroll
        for (int co = 0; co < COUT_TILE; ++co) {
            int gco = co_base + co;
            pwv[co] = (gco < Cout) ? ldg(pw_w + (int64_t)gco * Cin + c) : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int ow = ow0 + i;
            if (ow >= W) break;

            float dw = dbv;
            dw = fmaf(a0, w00, dw); dw = fmaf(a1, w01, dw); dw = fmaf(a2, w02, dw);
            dw = fmaf(b0, w10, dw); dw = fmaf(b1, w11, dw); dw = fmaf(b2, w12, dw);
            dw = fmaf(c0, w20, dw); dw = fmaf(c1, w21, dw); dw = fmaf(c2, w22, dw);

            #pragma unroll
            for (int co = 0; co < COUT_TILE; ++co) {
                acc[co][i] = fmaf(dw, pwv[co], acc[co][i]);
            }

            if (i != WPT - 1) {
                int nxt = iw_base + 3 + i;
                a0 = a1; a1 = a2; a2 = ld_x(ih0 + 0, nxt);
                b0 = b1; b1 = b2; b2 = ld_x(ih0 + 1, nxt);
                c0 = c1; c1 = c2; c2 = ld_x(ih0 + 2, nxt);
            }
        }
    }

    // Now each thread holds a partial sum over its Cin-stride subset.
    // Reduce across threads in the block for each (co,i) and pixel. We reduce per pixel by using
    // shared memory indexed by [co][thread][i], then sum on a subset of threads.
    extern __shared__ float smem[];
    // layout: [COUT_TILE][WPT][nthreads]
    int stride_threads = nthreads;
    int stride_wpt = stride_threads;
    int stride_co = WPT * stride_wpt;

    // write partials
    #pragma unroll
    for (int co = 0; co < COUT_TILE; ++co) {
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            smem[co * stride_co + i * stride_wpt + tid] = acc[co][i];
        }
    }
    __syncthreads();

    // first warp of each (ty,tx) pixel doesn't exist; reduction is across all threads for same pixel,
    // but threads correspond to different pixels. So we must reduce independently per pixel-thread.
    // Instead, we perform reduction across Cin-partition threads (all threads) for each pixel-thread
    // which is impossible because each thread owns different pixel. Therefore, ensure correctness by
    // mapping nthreads such that each pixel is computed by all threads: not the case.
    // We correct that by: each thread computed over different Cin subsets but for SAME pixel coords
    // only if all threads share same (oh,ow). They do not. So we must store per-thread outputs directly
    // WITHOUT reduction across threads. That means each thread must cover ALL Cin. To keep occupancy,
    // we reduce nthreads used for Cin-parallelism: do Cin loop on all threads identically (no partition).
    // This kernel version is thus not used.

    // NOTE: unreachable, kept for compilation completeness
    (void)stride_threads; (void)stride_wpt; (void)stride_co;
}

template<int BX, int BY, int WPT, int COUT_TILE, bool HAS_DW_BIAS, bool HAS_PW_BIAS>
__global__ __launch_bounds__(BX*BY, 3)
void fused_dw3x3_pw1x1_singlepass(
    const float* __restrict__ x,        // [N,Cin,H,W]
    const float* __restrict__ dw_w,     // [Cin,1,3,3]
    const float* __restrict__ dw_b,     // [Cin] or nullptr
    const float* __restrict__ pw_w,     // [Cout,Cin]
    const float* __restrict__ pw_b,     // [Cout] or nullptr
    float* __restrict__ y,              // [N,Cout,H,W]
    int N, int Cin, int H, int W,
    int Cout
) {
    const int HW = H * W;
    int ct_tiles = (Cout + COUT_TILE - 1) / COUT_TILE;
    int bz = (int)blockIdx.z;
    int n = bz / ct_tiles;
    int ct = bz - n * ct_tiles;
    if (n >= N) return;
    int co_base = ct * COUT_TILE;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    int oh = (int)blockIdx.y * BY + ty;
    int ow0 = ((int)blockIdx.x * BX + tx) * WPT;

    if (oh >= H || ow0 >= W) return;

    float acc[COUT_TILE][WPT];
    #pragma unroll
    for (int co = 0; co < COUT_TILE; ++co) {
        float b0 = 0.f;
        if constexpr (HAS_PW_BIAS) {
            int gco = co_base + co;
            if (gco < Cout) b0 = ldg(pw_b + gco);
        }
        #pragma unroll
        for (int i = 0; i < WPT; ++i) acc[co][i] = b0;
    }

    const float* x_n = x + (int64_t)n * Cin * (int64_t)HW;
    float* y_n = y + (int64_t)n * Cout * (int64_t)HW;

    // Full Cin reduction per thread (correct)
    for (int c = 0; c < Cin; ++c) {
        const float* x_nc = x_n + (int64_t)c * HW;

        const float* dwc = dw_w + c * 9;
        float w00 = ldg(dwc + 0), w01 = ldg(dwc + 1), w02 = ldg(dwc + 2);
        float w10 = ldg(dwc + 3), w11 = ldg(dwc + 4), w12 = ldg(dwc + 5);
        float w20 = ldg(dwc + 6), w21 = ldg(dwc + 7), w22 = ldg(dwc + 8);
        float dbv = 0.0f;
        if constexpr (HAS_DW_BIAS) dbv = ldg(dw_b + c);

        int ih0 = oh - 1;

        auto ld_x = [&](int ih, int iw)->float {
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                return ldg(x_nc + ih * W + iw);
            }
            return 0.0f;
        };

        int iw_base = ow0 - 1;

        float a0 = ld_x(ih0 + 0, iw_base + 0);
        float a1 = ld_x(ih0 + 0, iw_base + 1);
        float a2 = ld_x(ih0 + 0, iw_base + 2);

        float b0 = ld_x(ih0 + 1, iw_base + 0);
        float b1 = ld_x(ih0 + 1, iw_base + 1);
        float b2 = ld_x(ih0 + 1, iw_base + 2);

        float c0 = ld_x(ih0 + 2, iw_base + 0);
        float c1 = ld_x(ih0 + 2, iw_base + 1);
        float c2 = ld_x(ih0 + 2, iw_base + 2);

        float pwv[COUT_TILE];
        #pragma unroll
        for (int co = 0; co < COUT_TILE; ++co) {
            int gco = co_base + co;
            pwv[co] = (gco < Cout) ? ldg(pw_w + (int64_t)gco * Cin + c) : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int ow = ow0 + i;
            if (ow >= W) break;

            float dw = dbv;
            dw = fmaf(a0, w00, dw); dw = fmaf(a1, w01, dw); dw = fmaf(a2, w02, dw);
            dw = fmaf(b0, w10, dw); dw = fmaf(b1, w11, dw); dw = fmaf(b2, w12, dw);
            dw = fmaf(c0, w20, dw); dw = fmaf(c1, w21, dw); dw = fmaf(c2, w22, dw);

            #pragma unroll
            for (int co = 0; co < COUT_TILE; ++co) {
                acc[co][i] = fmaf(dw, pwv[co], acc[co][i]);
            }

            if (i != WPT - 1) {
                int nxt = iw_base + 3 + i;
                a0 = a1; a1 = a2; a2 = ld_x(ih0 + 0, nxt);
                b0 = b1; b1 = b2; b2 = ld_x(ih0 + 1, nxt);
                c0 = c1; c1 = c2; c2 = ld_x(ih0 + 2, nxt);
            }
        }
    }

    // store
    int hw_base = oh * W + ow0;
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        int ow = ow0 + i;
        if (ow >= W) break;
        int hw = oh * W + ow;
        #pragma unroll
        for (int co = 0; co < COUT_TILE; ++co) {
            int gco = co_base + co;
            if (gco < Cout) {
                y_n[(int64_t)gco * HW + hw] = acc[co][i];
            }
        }
    }
}

torch::Tensor fused_depthwise_pointwise_forward_cuda(
    torch::Tensor x,                       // [N,Cin,H,W]
    torch::Tensor dw_weight,               // [Cin,1,3,3]
    c10::optional<torch::Tensor> dw_bias,
    torch::Tensor pw_weight,               // [Cout,Cin,1,1]
    c10::optional<torch::Tensor> pw_bias,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
) {
    CHECK_INPUT(x);
    CHECK_INPUT(dw_weight);
    CHECK_INPUT(pw_weight);

    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(dw_weight.dim() == 4 && dw_weight.size(1) == 1, "dw_weight must be [Cin,1,kH,kW]");
    TORCH_CHECK(pw_weight.dim() == 4 && pw_weight.size(2) == 1 && pw_weight.size(3) == 1, "pw_weight must be [Cout,Cin,1,1]");

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(dw_weight.size(0) == Cin, "dw_weight Cin mismatch");
    TORCH_CHECK(pw_weight.size(1) == Cin, "pw_weight Cin mismatch");
    int64_t Cout = pw_weight.size(0);

    int64_t kH = dw_weight.size(2), kW = dw_weight.size(3);
    TORCH_CHECK(kH == 3 && kW == 3, "fused kernel supports dw k=3 only");
    TORCH_CHECK(sH == 1 && sW == 1, "fused kernel supports stride=1 only");
    TORCH_CHECK(pH == 1 && pW == 1, "fused kernel supports padding=1 only");
    TORCH_CHECK(dH == 1 && dW == 1, "fused kernel supports dilation=1 only");

    auto y = torch::empty({N, Cout, H, W}, x.options());

    const float* dw_bptr = nullptr;
    const float* pw_bptr = nullptr;
    bool has_dw_bias = false;
    bool has_pw_bias = false;

    torch::Tensor dw_b_t, pw_b_t;
    if (dw_bias.has_value() && dw_bias.value().defined()) {
        dw_b_t = dw_bias.value();
        CHECK_INPUT(dw_b_t);
        TORCH_CHECK(dw_b_t.dim() == 1 && dw_b_t.size(0) == Cin, "dw_bias must be [Cin]");
        dw_bptr = dw_b_t.data_ptr<float>();
        has_dw_bias = true;
    }
    if (pw_bias.has_value() && pw_bias.value().defined()) {
        pw_b_t = pw_bias.value();
        CHECK_INPUT(pw_b_t);
        TORCH_CHECK(pw_b_t.dim() == 1 && pw_b_t.size(0) == Cout, "pw_bias must be [Cout]");
        pw_bptr = pw_b_t.data_ptr<float>();
        has_pw_bias = true;
    }

    constexpr int BX = 16;
    constexpr int BY = 4;
    constexpr int WPT = 4;        // 64 pixels per block row in x (16*4)
    constexpr int COUT_TILE = 8;  // keep registers bounded

    dim3 block(BX, BY, 1);
    int ct_tiles = (int)((Cout + COUT_TILE - 1) / COUT_TILE);
    dim3 grid(
        (unsigned)((W + (BX * WPT) - 1) / (BX * WPT)),
        (unsigned)((H + BY - 1) / BY),
        (unsigned)(N * ct_tiles)
    );

    const float* pw_w2d = pw_weight.data_ptr<float>(); // treat contiguous as [Cout,Cin]

    auto stream = at::cuda::getDefaultCUDAStream();

    if (has_dw_bias && has_pw_bias) {
        fused_dw3x3_pw1x1_singlepass<BX, BY, WPT, COUT_TILE, true, true><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            dw_weight.data_ptr<float>(),
            dw_bptr,
            pw_w2d,
            pw_bptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W, (int)Cout
        );
    } else if (has_dw_bias && !has_pw_bias) {
        fused_dw3x3_pw1x1_singlepass<BX, BY, WPT, COUT_TILE, true, false><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            dw_weight.data_ptr<float>(),
            dw_bptr,
            pw_w2d,
            nullptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W, (int)Cout
        );
    } else if (!has_dw_bias && has_pw_bias) {
        fused_dw3x3_pw1x1_singlepass<BX, BY, WPT, COUT_TILE, false, true><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            dw_weight.data_ptr<float>(),
            nullptr,
            pw_w2d,
            pw_bptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W, (int)Cout
        );
    } else {
        fused_dw3x3_pw1x1_singlepass<BX, BY, WPT, COUT_TILE, false, false><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            dw_weight.data_ptr<float>(),
            nullptr,
            pw_w2d,
            nullptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W, (int)Cout
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
);

torch::Tensor pointwise_conv2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt
);

torch::Tensor fused_depthwise_pointwise_forward_cuda(
    torch::Tensor x,
    torch::Tensor dw_weight,
    c10::optional<torch::Tensor> dw_bias,
    torch::Tensor pw_weight,
    c10::optional<torch::Tensor> pw_bias,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_depthwise_separable2d_v7",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "depthwise_conv2d_forward_cuda",
        "pointwise_conv2d_forward_cuda",
        "fused_depthwise_pointwise_forward_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Depthwise-separable Conv2d replacement (forward-only) using custom CUDA kernels.

    Fast path: fused depthwise(3x3, stride=1, pad=1, dilation=1) + pointwise(1x1).
    Fallback: separate depthwise + pointwise kernels for other configurations.

    Constraints: CUDA input; float32 contiguous tensors (coerced in forward).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)

        self.sH = int(stride)
        self.sW = int(stride)
        self.pH = int(padding)
        self.pW = int(padding)
        self.dH = int(dilation)
        self.dW = int(dilation)

        self.depthwise_weight = nn.Parameter(torch.empty(self.in_channels, 1, self.kernel_size, self.kernel_size))
        self.pointwise_weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, 1, 1))

        if bias:
            self.depthwise_bias = nn.Parameter(torch.zeros(self.in_channels))
            self.pointwise_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.depthwise_bias = None
            self.pointwise_bias = None

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.depthwise_weight, a=5**0.5)
            nn.init.kaiming_uniform_(self.pointwise_weight, a=5**0.5)

        self.custom_ops = custom_ops_lib

    @staticmethod
    def _as_cuda_f32_contig(t: torch.Tensor, device: torch.device) -> torch.Tensor:
        if t is None:
            return None
        if t.device != device:
            t = t.to(device=device)
        if t.dtype != torch.float32:
            t = t.float()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")

        x = self._as_cuda_f32_contig(x, x.device)
        dw_w = self._as_cuda_f32_contig(self.depthwise_weight, x.device)
        pw_w = self._as_cuda_f32_contig(self.pointwise_weight, x.device)

        dw_b_opt = None
        pw_b_opt = None
        if self.depthwise_bias is not None:
            dw_b_opt = self._as_cuda_f32_contig(self.depthwise_bias, x.device)
            pw_b_opt = self._as_cuda_f32_contig(self.pointwise_bias, x.device)

        if (
            self.kernel_size == 3
            and self.sH == 1 and self.sW == 1
            and self.pH == 1 and self.pW == 1
            and self.dH == 1 and self.dW == 1
        ):
            return self.custom_ops.fused_depthwise_pointwise_forward_cuda(
                x, dw_w, dw_b_opt, pw_w, pw_b_opt,
                self.sH, self.sW, self.pH, self.pW, self.dH, self.dW
            )

        y_dw = self.custom_ops.depthwise_conv2d_forward_cuda(
            x, dw_w, dw_b_opt,
            self.sH, self.sW,
            self.pH, self.pW,
            self.dH, self.dW,
        )
        y = self.custom_ops.pointwise_conv2d_forward_cuda(
            y_dw, pw_w, pw_b_opt
        )
        return y