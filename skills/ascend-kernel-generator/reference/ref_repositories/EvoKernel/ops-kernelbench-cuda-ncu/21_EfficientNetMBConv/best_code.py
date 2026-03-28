import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# -----------------------------------------------------------------------------
# v3: Replace block-wide shared-memory reduction with warp-reduce + atomicAdd
#     into a global partial buffer, then a tiny finalize kernel applies BN(P)
#     and writes output.
# Also: put small BN vectors in __constant__ (safe size) and cache uploads.
#
# Specialization: Cin=112, hidden=672, Cout=192, K=5, stride=2, pad=2
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static constexpr int CIN = 112;
static constexpr int HIDDEN = 672;
static constexpr int COUT = 192;
static constexpr int K = 5;
static constexpr int STRIDE = 2;
static constexpr int PAD = 2;
static constexpr int TAP = K*K;

static constexpr int OC_TILE = 8;           // keep small vector acc
static constexpr int BLOCK_THREADS = 256;   // more warps, better hiding; warp-reduce path

__constant__ float c_aE[HIDDEN];
__constant__ float c_bE[HIDDEN];
__constant__ float c_aDW[HIDDEN];
__constant__ float c_bDW[HIDDEN];
__constant__ float c_aP[COUT];
__constant__ float c_bP[COUT];

__device__ __forceinline__ float relu6(float x) {
    x = x > 0.f ? x : 0.f;
    return x < 6.f ? x : 6.f;
}

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Compute partial sums for projection BEFORE BN(P):
// partial: [pixels, oc_tiles, OC_TILE] flattened as [pixels * num_tiles * OC_TILE]
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void mbconv_partial_fwd(
    const float* __restrict__ x,    // [N,112,Hin,Win] NCHW
    const float* __restrict__ wE,   // [672,112]
    const float* __restrict__ wDW,  // [672,25]
    const float* __restrict__ wP,   // [192,672]
    float* __restrict__ partial,    // [pixels, num_tiles, OC_TILE]
    int N, int Hin, int Win, int Hout, int Wout, int num_tiles
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int pix = (int)blockIdx.x;
    const int oc_tile = (int)blockIdx.y;
    const int oc0 = oc_tile * OC_TILE;
    if (oc0 >= COUT) return;

    const int ow = pix % Wout;
    const int t1 = pix / Wout;
    const int oh = t1 % Hout;
    const int n  = t1 / Hout;

    const int in_y0 = oh * STRIDE - PAD;
    const int in_x0 = ow * STRIDE - PAD;

    const int in_spatial = Hin * Win;

    float acc[OC_TILE];
    #pragma unroll
    for (int i = 0; i < OC_TILE; ++i) acc[i] = 0.f;

    // Loop hidden channels in a strided fashion.
    // Keep expand window in registers (25 floats) but limit live ranges:
    // compute ewin, then immediately dw, then project.
    for (int h = tid; h < HIDDEN; h += BLOCK_THREADS) {
        float ewin[TAP];

        const float* __restrict__ wE_h = wE + h * CIN;
        const float aEh = c_aE[h];
        const float bEh = c_bE[h];

        #pragma unroll
        for (int ky = 0; ky < K; ++ky) {
            const int iy = in_y0 + ky;
            const bool y_in = ((unsigned)iy < (unsigned)Hin);
            #pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                const int ix = in_x0 + kx;
                const bool inb = y_in && ((unsigned)ix < (unsigned)Win);
                float e = 0.f;
                if (inb) {
                    const int x_base = ((n * CIN) * Hin + iy) * Win + ix; // x[n,0,iy,ix]
                    float sum = 0.f;
                    // CIN=112: keep unroll moderate to avoid reg explosion
                    #pragma unroll 4
                    for (int ic = 0; ic < CIN; ++ic) {
                        float xv = x[x_base + ic * in_spatial];
                        float wv = wE_h[ic];
                        sum = fmaf(xv, wv, sum);
                    }
                    e = relu6(fmaf(sum, aEh, bEh));
                }
                ewin[ky * K + kx] = e;
            }
        }

        const float* __restrict__ wdw_h = wDW + h * TAP;
        float dw_sum = 0.f;
        #pragma unroll
        for (int t = 0; t < TAP; ++t) {
            dw_sum = fmaf(ewin[t], wdw_h[t], dw_sum);
        }
        float dw = relu6(fmaf(dw_sum, c_aDW[h], c_bDW[h]));

        // Projection: oc contiguous within tile.
        // wP laid out [COUT,HIDDEN] row-major => wP[oc*HIDDEN + h]
        #pragma unroll
        for (int i = 0; i < OC_TILE; ++i) {
            const int oc = oc0 + i;
            if (oc < COUT) {
                float wp = wP[oc * HIDDEN + h];
                acc[i] = fmaf(dw, wp, acc[i]);
            }
        }
    }

    // Warp-reduce each acc[i] within warp, then atomicAdd lane0 into partial.
    // One atomic per warp per i => (BLOCK_THREADS/32)*OC_TILE atomics per block.
    #pragma unroll
    for (int i = 0; i < OC_TILE; ++i) {
        float v = warp_reduce_sum(acc[i]);
        if (lane == 0) {
            const long out_idx = ((long)pix * (long)num_tiles + (long)oc_tile) * (long)OC_TILE + (long)i;
            atomicAdd(&partial[out_idx], v);
        }
    }
}

// Finalize: apply BN(P) and write y in NCHW.
// Each thread handles one element in partial and writes one y.
__global__ void mbconv_finalize_fwd(
    const float* __restrict__ partial, // [pixels, num_tiles, OC_TILE]
    float* __restrict__ y,             // [N,192,Hout,Wout]
    int pixels, int N, int Hout, int Wout, int num_tiles
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = pixels * num_tiles * OC_TILE;
    if (idx >= total) return;

    int i = idx % OC_TILE;
    int tmp = idx / OC_TILE;
    int oc_tile = tmp % num_tiles;
    int pix = tmp / num_tiles;

    int oc = oc_tile * OC_TILE + i;
    if (oc >= COUT) return;

    float v = partial[idx];
    v = fmaf(v, c_aP[oc], c_bP[oc]);

    int ow = pix % Wout;
    int t1 = pix / Wout;
    int oh = t1 % Hout;
    int n  = t1 / Hout;

    int out_spatial = Hout * Wout;
    long out_base = ((long)n * (long)COUT) * (long)out_spatial + (long)oh * (long)Wout + (long)ow;
    y[out_base + (long)oc * (long)out_spatial] = v;
}

static void upload_const_vec(const torch::Tensor& t, const void* sym, size_t bytes) {
    // device-to-device, synchronous is fine for inference path and avoids stream API issues
    C10_CUDA_CHECK(cudaMemcpyToSymbol(sym, t.data_ptr<float>(), bytes, 0, cudaMemcpyDeviceToDevice));
}

static void upload_all_constants(
    const torch::Tensor& aE, const torch::Tensor& bE,
    const torch::Tensor& aDW, const torch::Tensor& bDW,
    const torch::Tensor& aP, const torch::Tensor& bP
) {
    upload_const_vec(aE,  c_aE,  sizeof(float) * HIDDEN);
    upload_const_vec(bE,  c_bE,  sizeof(float) * HIDDEN);
    upload_const_vec(aDW, c_aDW, sizeof(float) * HIDDEN);
    upload_const_vec(bDW, c_bDW, sizeof(float) * HIDDEN);
    upload_const_vec(aP,  c_aP,  sizeof(float) * COUT);
    upload_const_vec(bP,  c_bP,  sizeof(float) * COUT);
}

torch::Tensor efficient_net_mb_conv_forward_cuda(
    torch::Tensor x,          // [N,112,H,W]
    torch::Tensor wE,         // [672,112]
    torch::Tensor aE,         // [672]
    torch::Tensor bE,         // [672]
    torch::Tensor wDW,        // [672,25]
    torch::Tensor aDW,        // [672]
    torch::Tensor bDW,        // [672]
    torch::Tensor wP,         // [192,672]
    torch::Tensor aP,         // [192]
    torch::Tensor bP          // [192]
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x);
    CHECK_CUDA(wE); CHECK_CONTIGUOUS(wE); CHECK_FLOAT(wE);
    CHECK_CUDA(aE); CHECK_CONTIGUOUS(aE); CHECK_FLOAT(aE);
    CHECK_CUDA(bE); CHECK_CONTIGUOUS(bE); CHECK_FLOAT(bE);
    CHECK_CUDA(wDW); CHECK_CONTIGUOUS(wDW); CHECK_FLOAT(wDW);
    CHECK_CUDA(aDW); CHECK_CONTIGUOUS(aDW); CHECK_FLOAT(aDW);
    CHECK_CUDA(bDW); CHECK_CONTIGUOUS(bDW); CHECK_FLOAT(bDW);
    CHECK_CUDA(wP); CHECK_CONTIGUOUS(wP); CHECK_FLOAT(wP);
    CHECK_CUDA(aP); CHECK_CONTIGUOUS(aP); CHECK_FLOAT(aP);
    CHECK_CUDA(bP); CHECK_CONTIGUOUS(bP); CHECK_FLOAT(bP);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK((int)x.size(1) == CIN, "Cin must be 112");
    TORCH_CHECK(wE.dim() == 2 && (int)wE.size(0) == HIDDEN && (int)wE.size(1) == CIN, "wE must be [672,112]");
    TORCH_CHECK(wDW.dim() == 2 && (int)wDW.size(0) == HIDDEN && (int)wDW.size(1) == TAP, "wDW must be [672,25]");
    TORCH_CHECK(wP.dim() == 2 && (int)wP.size(0) == COUT && (int)wP.size(1) == HIDDEN, "wP must be [192,672]");
    TORCH_CHECK(aE.numel() == HIDDEN && bE.numel() == HIDDEN, "aE/bE must be [672]");
    TORCH_CHECK(aDW.numel() == HIDDEN && bDW.numel() == HIDDEN, "aDW/bDW must be [672]");
    TORCH_CHECK(aP.numel() == COUT && bP.numel() == COUT, "aP/bP must be [192]");

    const int N = (int)x.size(0);
    const int Hin = (int)x.size(2);
    const int Win = (int)x.size(3);
    const int Hout = (Hin + 2 * PAD - K) / STRIDE + 1;
    const int Wout = (Win + 2 * PAD - K) / STRIDE + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    // Upload constants every call (simple + robust). In eval the tensors are stable anyway.
    upload_all_constants(aE, bE, aDW, bDW, aP, bP);

    auto y = torch::empty({N, COUT, Hout, Wout}, x.options());

    long pixels = (long)N * (long)Hout * (long)Wout;
    const int num_tiles = (COUT + OC_TILE - 1) / OC_TILE;

    // Partial buffer (zeroed) for atomic accumulation
    auto partial = torch::zeros({(long)pixels, (long)num_tiles, (long)OC_TILE}, x.options());

    dim3 block(BLOCK_THREADS);
    dim3 grid((unsigned int)pixels, (unsigned int)num_tiles, 1);

    mbconv_partial_fwd<<<grid, block>>>(
        x.data_ptr<float>(),
        wE.data_ptr<float>(),
        wDW.data_ptr<float>(),
        wP.data_ptr<float>(),
        partial.data_ptr<float>(),
        N, Hin, Win, Hout, Wout, num_tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Finalize
    int total = (int)(pixels * (long)num_tiles * (long)OC_TILE);
    int tblock = 256;
    int tgrid = (total + tblock - 1) / tblock;
    mbconv_finalize_fwd<<<tgrid, tblock>>>(
        partial.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)pixels, N, Hout, Wout, num_tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor efficient_net_mb_conv_forward_cuda(
    torch::Tensor x,
    torch::Tensor wE,
    torch::Tensor aE,
    torch::Tensor bE,
    torch::Tensor wDW,
    torch::Tensor aDW,
    torch::Tensor bDW,
    torch::Tensor wP,
    torch::Tensor aP,
    torch::Tensor bP
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_efficient_net_mb_conv_v3_warpatomic",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["efficient_net_mb_conv_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    @staticmethod
    def _bn_alpha_beta_infer(bn: nn.BatchNorm2d, device, dtype):
        gamma = bn.weight.to(device=device, dtype=dtype)
        beta = bn.bias.to(device=device, dtype=dtype)
        mean = bn.running_mean.to(device=device, dtype=dtype)
        var = bn.running_var.to(device=device, dtype=dtype)
        invstd = torch.rsqrt(var + bn.eps)
        alpha = gamma * invstd
        beta2 = beta - mean * alpha
        return alpha.contiguous(), beta2.contiguous()

    def _fallback(self, x):
        identity = x
        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = x + identity
        return x

    def forward(self, x):
        if (
            self.training
            or (not x.is_cuda)
            or (x.dtype != torch.float32)
            or (not x.is_contiguous())
            or self.use_residual
            or (self.in_channels != 112)
            or (self.hidden_dim != 672)
            or (self.out_channels != 192)
            or (self.kernel_size != 5)
            or (self.stride != 2)
            or (self.expand_ratio != 6)
        ):
            return self._fallback(x)

        if not hasattr(self, "expand_conv"):
            return self._fallback(x)

        convE: nn.Conv2d = self.expand_conv[0]
        bnE: nn.BatchNorm2d = self.expand_conv[1]
        convDW: nn.Conv2d = self.depthwise_conv[0]
        bnDW: nn.BatchNorm2d = self.depthwise_conv[1]
        convP: nn.Conv2d = self.project_conv[0]
        bnP: nn.BatchNorm2d = self.project_conv[1]

        if convE.kernel_size != (1, 1) or convE.stride != (1, 1) or convE.padding != (0, 0):
            return self._fallback(x)
        if convDW.kernel_size != (5, 5) or convDW.stride != (2, 2) or convDW.padding != (2, 2) or convDW.groups != 672:
            return self._fallback(x)
        if convP.kernel_size != (1, 1) or convP.stride != (1, 1) or convP.padding != (0, 0):
            return self._fallback(x)

        if not (convE.weight.is_cuda and convDW.weight.is_cuda and convP.weight.is_cuda):
            return self._fallback(x)

        device = x.device
        dtype = x.dtype

        wE = convE.weight.contiguous().view(672, 112)
        wDW = convDW.weight.contiguous().view(672, 25)
        wP = convP.weight.contiguous().view(192, 672)

        aE, bE = self._bn_alpha_beta_infer(bnE, device, dtype)
        aDW, bDW = self._bn_alpha_beta_infer(bnDW, device, dtype)
        aP, bP = self._bn_alpha_beta_infer(bnP, device, dtype)

        return custom_ops_lib.efficient_net_mb_conv_forward_cuda(
            x,
            wE, aE, bE,
            wDW, aDW, bDW,
            wP, aP, bP,
        )