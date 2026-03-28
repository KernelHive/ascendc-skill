import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# Optimized fused CUDA op:
#   ConvTranspose3d (FAST PATH: K=3, stride=2, pad=1, groups=1, dilation=1, out_pad=0)
#     - gather (no atomics)
#     - optional bias fused into conv
#     - output-channel tiling with shared-memory staged weights
#   -> BatchNorm3d (training-style batch stats; affine gamma/beta; no running stats update)
#   -> subtract spatial mean per (N,C) AFTER BN (keepdim behavior)
#
# Fusions vs baseline:
#   - fuse conv + bias
#   - remove mean_nc materialization and subtract pass by fusing BN affine + spatial mean subtract
#
# Supports:
#   - CUDA float32 only
#   - contiguous NCDHW
#   - weight [Cin,Cout,3,3,3] for fast path, otherwise fallback to baseline scatter+atomics
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

static __forceinline__ __device__ float warp_reduce_sum_fullmask(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// ------------------------------------------------------------
// Generic fallback: scatter+atomic convT (same as baseline)
// ------------------------------------------------------------
__global__ void convT3d_scatter_strided_kernel(
    const float* __restrict__ x,     // [N,Cin,Di,Hi,Wi]
    const float* __restrict__ w,     // [Cin,Cout,kD,kH,kW]
    float* __restrict__ y,           // [N,Cout,Do,Ho,Wo] pre-zeroed
    int N, int Cin, int Cout,
    int Di, int Hi, int Wi,
    int kD, int kH, int kW,
    int stride, int padding,
    int Do, int Ho, int Wo
){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Cin * Di * Hi * Wi;
    if (idx >= total) return;

    int64_t t = idx;
    int iw = (int)(t % Wi); t /= Wi;
    int ih = (int)(t % Hi); t /= Hi;
    int id = (int)(t % Di); t /= Di;
    int ci = (int)(t % Cin); t /= Cin;
    int n  = (int)t;

    float xv = x[idx];

    int64_t w_ci_base = (int64_t)ci * Cout * (int64_t)kD * kH * kW;
    int64_t y_n_base  = (int64_t)n  * Cout * (int64_t)Do * Ho * Wo;
    int64_t stride_y_co = (int64_t)Do * Ho * Wo;
    int64_t stride_w_co = (int64_t)kD * kH * kW;

    for (int kd = 0; kd < kD; ++kd) {
        int od = id * stride - padding + kd;
        if ((unsigned)od >= (unsigned)Do) continue;
        for (int kh = 0; kh < kH; ++kh) {
            int oh = ih * stride - padding + kh;
            if ((unsigned)oh >= (unsigned)Ho) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int ow = iw * stride - padding + kw;
                if ((unsigned)ow >= (unsigned)Wo) continue;

                int64_t w_k = ((int64_t)kd * kH + kh) * (int64_t)kW + kw;
                int64_t w_base = w_ci_base + w_k;

                int64_t y_spatial = ((int64_t)od * Ho + oh) * (int64_t)Wo + ow;
                int64_t y_base = y_n_base + y_spatial;

                for (int co = 0; co < Cout; ++co) {
                    float wv = w[w_base + (int64_t)co * stride_w_co];
                    atomicAdd(&y[y_base + (int64_t)co * stride_y_co], xv * wv);
                }
            }
        }
    }
}

__global__ void add_bias_kernel(
    float* __restrict__ y,            // [N,Cout,Do,Ho,Wo]
    const float* __restrict__ bias,   // [Cout]
    int64_t total,
    int Cout,
    int Do, int Ho, int Wo
){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t t = idx;
    (void)(t % Wo); t /= Wo;
    (void)(t % Ho); t /= Ho;
    (void)(t % Do); t /= Do;
    int co = (int)(t % Cout);

    y[idx] += bias[co];
}

// ------------------------------------------------------------
// FAST ConvTranspose3d gather specialized for:
//   K=3, stride=2, pad=1, dilation=1, output_padding=0
// Weight layout: [Cin, Cout, 3,3,3]
// Output dims: Do=2*Di-1, Ho=2*Hi-1, Wo=2*Wi-1
//
// Channel tiling: COTILE output channels per block, staged weights into shared memory
// Grid: blockIdx.y = n*CoutTiles + cotile
//       blockIdx.x over spatial tiles (linear over Do*Ho*Wo)
// ------------------------------------------------------------
template<int COTILE, int TILE_ELEMS>
__global__ __launch_bounds__(256, 2)
void convT3d_gather_s2p1k3_cotile_kernel(
    const float* __restrict__ x,      // [N,Cin,Di,Hi,Wi]
    const float* __restrict__ w,      // [Cin,Cout,3,3,3]
    const float* __restrict__ bias,   // [Cout] or nullptr
    float* __restrict__ y,            // [N,Cout,Do,Ho,Wo]
    int N, int Cin, int Cout,
    int Di, int Hi, int Wi,
    int Do, int Ho, int Wo,
    int cout_tiles
){
    // spatial linear index base for this block
    int64_t spatial = (int64_t)Do * Ho * Wo;
    int64_t base_sp = (int64_t)blockIdx.x * (int64_t)TILE_ELEMS;
    int lane = threadIdx.x;

    // which (n, co tile)
    int block_y = (int)blockIdx.y;
    int n = block_y / cout_tiles;
    int cot = block_y - n * cout_tiles;
    int co0 = cot * COTILE;

    if (n >= N) return;

    // stage weights for this (co0..co0+COTILE-1) tile into shared memory
    // Layout: sh_w[(ci * 27 + k) * COTILE + tco]
    extern __shared__ float shmem[];
    float* sh_w = shmem;

    int64_t w_ci_stride = (int64_t)Cout * 27;
    int64_t w_co_stride = 27;

    int total_w = Cin * 27 * COTILE;
    for (int i = lane; i < total_w; i += blockDim.x) {
        int tco = i % COTILE;
        int tmp = i / COTILE;
        int k = tmp % 27;
        int ci = tmp / 27;
        int co = co0 + tco;
        float v = 0.0f;
        if (co < Cout) {
            v = __ldg(w + (int64_t)ci * w_ci_stride + (int64_t)co * w_co_stride + k);
        }
        sh_w[i] = v;
    }
    __syncthreads();

    int64_t in_hw  = (int64_t)Hi * Wi;
    int64_t in_dhw = (int64_t)Di * in_hw;
    int64_t x_n_base = (int64_t)n * (int64_t)Cin * in_dhw;

    // each thread computes one (spatial element) and COTILE channels (small vector)
    // Use multiple "items" per thread via striding for better occupancy utilization
    for (int it = lane; it < TILE_ELEMS; it += blockDim.x) {
        int64_t sp = base_sp + it;
        if (sp >= spatial) continue;

        int ow = (int)(sp % Wo);
        int64_t tt = sp / Wo;
        int oh = (int)(tt % Ho);
        int od = (int)(tt / Ho);

        float acc[COTILE];
        #pragma unroll
        for (int i = 0; i < COTILE; ++i) acc[i] = 0.0f;

        // K=3 loops; parity inversion for stride=2, pad=1:
        // id = (od + 1 - kd)/2 must be integer
        #pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            int td = od + 1 - kd;
            if (td & 1) continue;
            int id = td >> 1;
            if ((unsigned)id >= (unsigned)Di) continue;

            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int th = oh + 1 - kh;
                if (th & 1) continue;
                int ih = th >> 1;
                if ((unsigned)ih >= (unsigned)Hi) continue;

                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int tw = ow + 1 - kw;
                    if (tw & 1) continue;
                    int iw = tw >> 1;
                    if ((unsigned)iw >= (unsigned)Wi) continue;

                    int64_t x_sp = (int64_t)id * in_hw + (int64_t)ih * Wi + iw;
                    int kidx = (kd * 9) + (kh * 3) + kw; // 0..26
                    // accumulate over Cin
                    #pragma unroll 1
                    for (int ci = 0; ci < Cin; ++ci) {
                        float xv = __ldg(x + x_n_base + (int64_t)ci * in_dhw + x_sp);
                        int sh_base = (ci * 27 + kidx) * COTILE;
                        #pragma unroll
                        for (int tco = 0; tco < COTILE; ++tco) {
                            float wv = sh_w[sh_base + tco];
                            acc[tco] = fmaf(xv, wv, acc[tco]);
                        }
                    }
                }
            }
        }

        int64_t y_base = ((int64_t)n * Cout * spatial) + sp;
        int64_t y_co_stride = spatial;

        #pragma unroll
        for (int tco = 0; tco < COTILE; ++tco) {
            int co = co0 + tco;
            if (co >= Cout) continue;
            float outv = acc[tco];
            if (bias) outv += __ldg(bias + co);
            y[y_base + (int64_t)co * y_co_stride] = outv;
        }
    }
}

// ------------------------------------------------------------
// BN stats (training) per channel over N*D*H*W using warp shuffles
// mean[c], var[c] (biased var like PyTorch training forward)
// ------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void bn_stats_warp_kernel(
    const float* __restrict__ x, // [N,C,D,H,W]
    float* __restrict__ mean,    // [C]
    float* __restrict__ var,     // [C]
    int N, int C, int D, int H, int W
){
    int c = (int)blockIdx.x;
    if (c >= C) return;

    int64_t spatial = (int64_t)D * H * W;
    int64_t M = (int64_t)N * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int64_t i = (int64_t)threadIdx.x; i < M; i += (int64_t)blockDim.x) {
        int n = (int)(i / spatial);
        int64_t s = i - (int64_t)n * spatial;
        int64_t off = ((int64_t)n * C + c) * spatial + s;
        float v = __ldg(x + off);
        sum += v;
        sumsq = fmaf(v, v, sumsq);
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    sum = warp_reduce_sum_fullmask(sum);
    sumsq = warp_reduce_sum_fullmask(sumsq);

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
        bsum = warp_reduce_sum_fullmask(bsum);
        bsumsq = warp_reduce_sum_fullmask(bsumsq);
        if (lane == 0) {
            float m = bsum / (float)M;
            float v = bsumsq / (float)M - m * m;
            mean[c] = m;
            var[c]  = v;
        }
    }
}

// ------------------------------------------------------------
// Fused BN affine + spatial mean subtract:
// For each (n,c): compute mean over spatial of BN(x), then write out = BN(x) - mean_spatial(BN(x)).
// This matches: y=BN(x); y - mean(y, dim=(2,3,4), keepdim=True)
//
// Two-phase within one block per (n,c):
//   1) reduce sum of BN(x) over spatial
//   2) second pass write BN(x) - mean
//
// Vectorize with float4 when aligned and spatial divisible by 4.
// ------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void bn_affine_and_spatial_mean_subtract_kernel(
    const float* __restrict__ x,      // [N,C,D,H,W]
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ var,    // [C]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ out,          // [N,C,D,H,W]
    int N, int C, int D, int H, int W,
    float eps
){
    int c = (int)blockIdx.x;
    int n = (int)blockIdx.y;
    if (c >= C || n >= N) return;

    int64_t spatial = (int64_t)D * H * W;
    int64_t base = ((int64_t)n * C + c) * spatial;

    float m = __ldg(mean + c);
    float v = __ldg(var + c);
    float inv = rsqrtf(v + eps);
    float g = __ldg(gamma + c);
    float be = __ldg(beta + c);

    // Pass 1: sum of BN(x)
    float local = 0.0f;

    // vectorized path
    const float* xp = x + base;
    bool vec_ok = (((uintptr_t)xp & 0xF) == 0) && ((spatial & 3) == 0);

    if (vec_ok) {
        const float4* x4 = (const float4*)xp;
        int64_t n4 = spatial >> 2;
        for (int64_t i = (int64_t)threadIdx.x; i < n4; i += (int64_t)blockDim.x) {
            float4 v4 = __ldg(x4 + i);
            float y0 = fmaf((v4.x - m) * inv, g, be);
            float y1 = fmaf((v4.y - m) * inv, g, be);
            float y2 = fmaf((v4.z - m) * inv, g, be);
            float y3 = fmaf((v4.w - m) * inv, g, be);
            local += (y0 + y1) + (y2 + y3);
        }
    } else {
        for (int64_t i = (int64_t)threadIdx.x; i < spatial; i += (int64_t)blockDim.x) {
            float xv = __ldg(xp + i);
            float yv = fmaf((xv - m) * inv, g, be);
            local += yv;
        }
    }

    // block reduction
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    local = warp_reduce_sum_fullmask(local);

    __shared__ float warp_acc[8];
    if (lane == 0) warp_acc[warp] = local;
    __syncthreads();

    float bsum = 0.0f;
    if (warp == 0) {
        bsum = (lane < (blockDim.x >> 5)) ? warp_acc[lane] : 0.0f;
        bsum = warp_reduce_sum_fullmask(bsum);
    }
    __shared__ float mean_sp;
    if (threadIdx.x == 0) mean_sp = bsum / (float)spatial;
    __syncthreads();
    float ms = mean_sp;

    // Pass 2: write BN(x) - spatial_mean
    float* op = out + base;
    if (vec_ok && (((uintptr_t)op & 0xF) == 0)) {
        const float4* x4 = (const float4*)xp;
        float4* o4 = (float4*)op;
        int64_t n4 = spatial >> 2;
        for (int64_t i = (int64_t)threadIdx.x; i < n4; i += (int64_t)blockDim.x) {
            float4 v4 = __ldg(x4 + i);
            float y0 = fmaf((v4.x - m) * inv, g, be) - ms;
            float y1 = fmaf((v4.y - m) * inv, g, be) - ms;
            float y2 = fmaf((v4.z - m) * inv, g, be) - ms;
            float y3 = fmaf((v4.w - m) * inv, g, be) - ms;
            o4[i] = make_float4(y0, y1, y2, y3);
        }
    } else {
        for (int64_t i = (int64_t)threadIdx.x; i < spatial; i += (int64_t)blockDim.x) {
            float xv = __ldg(xp + i);
            float yv = fmaf((xv - m) * inv, g, be) - ms;
            op[i] = yv;
        }
    }
}

// ------------------------------------------------------------
// Entry point
// ------------------------------------------------------------
torch::Tensor conv_transpose3d_batch_norm_subtract_cuda(
    torch::Tensor x,                  // [N,Cin,Di,Hi,Wi]
    torch::Tensor w,                  // [Cin,Cout,kD,kH,kW]
    c10::optional<torch::Tensor> b_opt,// [Cout] or None
    int64_t stride,
    int64_t padding,
    torch::Tensor bn_weight,          // [Cout]
    torch::Tensor bn_bias,            // [Cout]
    double eps
){
    CHECK_CUDA(x); CHECK_CUDA(w);
    CHECK_CUDA(bn_weight); CHECK_CUDA(bn_bias);
    CHECK_F32(x); CHECK_F32(w);
    CHECK_F32(bn_weight); CHECK_F32(bn_bias);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w);
    CHECK_CONTIGUOUS(bn_weight); CHECK_CONTIGUOUS(bn_bias);

    TORCH_CHECK(x.dim() == 5, "x must be [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be [Cin,Cout,kD,kH,kW]");
    TORCH_CHECK(bn_weight.dim() == 1 && bn_bias.dim() == 1, "bn params must be [Cout]");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    auto x_c = x;
    auto w_c = w;
    auto g_c = bn_weight;
    auto be_c = bn_bias;

    int64_t N   = x_c.size(0);
    int64_t Cin = x_c.size(1);
    int64_t Di  = x_c.size(2);
    int64_t Hi  = x_c.size(3);
    int64_t Wi  = x_c.size(4);

    int64_t wCin = w_c.size(0);
    int64_t Cout = w_c.size(1);
    int64_t kD   = w_c.size(2);
    int64_t kH   = w_c.size(3);
    int64_t kW   = w_c.size(4);

    TORCH_CHECK(wCin == Cin, "w.size(0) must match Cin");
    TORCH_CHECK(g_c.numel() == Cout && be_c.numel() == Cout, "bn params must have Cout elements");

    int64_t Do = (Di - 1) * stride - 2 * padding + kD;
    int64_t Ho = (Hi - 1) * stride - 2 * padding + kH;
    int64_t Wo = (Wi - 1) * stride - 2 * padding + kW;
    TORCH_CHECK(Do > 0 && Ho > 0 && Wo > 0, "computed output dims must be positive");

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();

    // Conv output (pre-BN)
    auto y = torch::empty({N, Cout, Do, Ho, Wo}, x_c.options());

    const float* bias_ptr = nullptr;
    torch::Tensor b_c;
    if (b_opt.has_value()) {
        b_c = b_opt.value();
        CHECK_CUDA(b_c);
        CHECK_F32(b_c);
        CHECK_CONTIGUOUS(b_c);
        TORCH_CHECK(b_c.dim() == 1 && b_c.numel() == Cout, "bias must be [Cout]");
        bias_ptr = b_c.data_ptr<float>();
    }

    bool fast = (stride == 2 && padding == 1 && kD == 3 && kH == 3 && kW == 3);

    if (fast) {
        // y is fully written (no need to zero)
        constexpr int COTILE = 8;
        constexpr int TILE_ELEMS = 1024; // per block spatial work chunk
        int cout_tiles = (int)((Cout + COTILE - 1) / COTILE);

        int64_t spatial = Do * Ho * Wo;
        int grid_x = (int)((spatial + TILE_ELEMS - 1) / TILE_ELEMS);
        dim3 grid(grid_x, (unsigned)(N * cout_tiles), 1);

        // shared weights size: Cin * 27 * COTILE floats
        size_t shmem = (size_t)Cin * 27 * COTILE * sizeof(float);

        const int threads = 256;
        convT3d_gather_s2p1k3_cotile_kernel<COTILE, TILE_ELEMS><<<grid, threads, shmem, stream>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Cout,
            (int)Di, (int)Hi, (int)Wi,
            (int)Do, (int)Ho, (int)Wo,
            cout_tiles
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // fallback: scatter+atomic into zeroed y then optional bias
        y.zero_();
        {
            int64_t total = N * Cin * Di * Hi * Wi;
            const int threads = 128;
            int blocks = (int)((total + threads - 1) / threads);
            convT3d_scatter_strided_kernel<<<blocks, threads, 0, stream>>>(
                x_c.data_ptr<float>(),
                w_c.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Cout,
                (int)Di, (int)Hi, (int)Wi,
                (int)kD, (int)kH, (int)kW,
                (int)stride, (int)padding,
                (int)Do, (int)Ho, (int)Wo
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (bias_ptr) {
            int64_t total = y.numel();
            const int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            add_bias_kernel<<<blocks, threads, 0, stream>>>(
                y.data_ptr<float>(),
                bias_ptr,
                total,
                (int)Cout,
                (int)Do, (int)Ho, (int)Wo
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // BN stats (training-style)
    auto mean = torch::empty({Cout}, x_c.options());
    auto var  = torch::empty({Cout}, x_c.options());
    {
        const int threads = 256;
        dim3 grid((unsigned)Cout);
        bn_stats_warp_kernel<<<grid, threads, 0, stream>>>(
            y.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            (int)N, (int)Cout, (int)Do, (int)Ho, (int)Wo
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Fused BN affine + spatial mean subtract (output overwrites y)
    {
        const int threads = 256;
        dim3 grid((unsigned)Cout, (unsigned)N, 1);
        bn_affine_and_spatial_mean_subtract_kernel<<<grid, threads, 0, stream>>>(
            y.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            g_c.data_ptr<float>(),
            be_c.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)Cout, (int)Do, (int)Ho, (int)Wo,
            (float)eps
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose3d_batch_norm_subtract_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride,
    int64_t padding,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convT3d_bn_subtract_gather_cotile_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_batch_norm_subtract_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Custom CUDA replacement for:
      ConvTranspose3d -> BatchNorm3d (training-style stats) -> subtract spatial mean

    Fast path:
      - float32 CUDA contiguous
      - kernel_size=3 (cubic), stride=2, padding=1
      - groups=1, dilation=1, output_padding=0

    Fallback:
      - generic scatter+atomic convT for other params (still fused with BN + mean subtract).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        if isinstance(kernel_size, (tuple, list)):
            kD = int(kernel_size[0])
            kH = int(kernel_size[1]) if len(kernel_size) > 1 else int(kernel_size[0])
            kW = int(kernel_size[2]) if len(kernel_size) > 2 else int(kernel_size[0])
        else:
            kD = kH = kW = int(kernel_size)
        if not (kD == kH == kW):
            raise ValueError("ModelNew expects cubic kernel_size.")
        self.kernel_size = int(kD)

        if isinstance(stride, (tuple, list)):
            s = int(stride[0])
        else:
            s = int(stride)
        self.stride = int(s)

        if isinstance(padding, (tuple, list)):
            p = int(padding[0])
        else:
            p = int(padding)
        self.padding = int(p)

        self.use_bias = bool(bias)
        self.eps = 1e-5

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

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.bn_weight = nn.Parameter(torch.ones(self.out_channels, dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew supports CUDA only")
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        dev = x.device
        w = self.weight if self.weight.device == dev else self.weight.to(dev)
        bw = self.bn_weight if self.bn_weight.device == dev else self.bn_weight.to(dev)
        bb = self.bn_bias if self.bn_bias.device == dev else self.bn_bias.to(dev)
        b = None
        if self.use_bias:
            b = self.bias if self.bias.device == dev else self.bias.to(dev)

        return self.custom_ops.conv_transpose3d_batch_norm_subtract_cuda(
            x,
            w.contiguous(),
            b.contiguous() if b is not None else None,
            int(self.stride),
            int(self.padding),
            bw.contiguous(),
            bb.contiguous(),
            float(self.eps),
        )