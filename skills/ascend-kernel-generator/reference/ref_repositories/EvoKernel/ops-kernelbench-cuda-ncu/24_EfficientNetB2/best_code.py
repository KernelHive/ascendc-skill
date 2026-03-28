import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension:
#  1) Specialized fast kernel for conv1: 3x3 s=2 p=1, Cin=3, Cout=32, fused BN(infer)+ReLU, NCHW
#     - Shared-memory input tiling
#     - Each thread computes 4 output channels for one output pixel (2x2 pixels per thread via loop)
#  2) Generic fallback kernels (baseline): k=3 or k=1, fused BN(infer)+ReLU
# -------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

#define CUDA_KERNEL_CHECK() do { \
  cudaError_t err = cudaGetLastError(); \
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err)); \
} while(0)

#if __CUDA_ARCH__ >= 350
  #define LDG(x) __ldg(x)
#else
  #define LDG(x) (*(x))
#endif

__device__ __forceinline__ float bn_relu(float v, float a, float b) {
    v = fmaf(v, a, b);
    return v > 0.0f ? v : 0.0f;
}

// ------------------------------------------------------------
// Generic fallback kernels (from baseline)
// ------------------------------------------------------------

__global__ __launch_bounds__(256, 2)
void conv2d_k3_bn_relu_nchw_generic(
    const float* __restrict__ x,     // [N,Cin,Hin,Win]
    const float* __restrict__ w,     // [Cout,Cin,3,3]
    const float* __restrict__ alpha, // [Cout]
    const float* __restrict__ beta,  // [Cout]
    float* __restrict__ y,           // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride, int pad
) {
    int nc = (int)blockIdx.y;
    int n = nc / Cout;
    int oc = nc - n * Cout;
    if (n >= N) return;

    float a = LDG(alpha + oc);
    float b = LDG(beta + oc);

    int spatial = Hout * Wout;
    int base_sp = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

    int w_oc_base = oc * Cin * 9;

    for (int sp = base_sp; sp < spatial; sp += (int)gridDim.x * (int)blockDim.x) {
        int oh = sp / Wout;
        int ow = sp - oh * Wout;

        int in_y0 = oh * stride - pad;
        int in_x0 = ow * stride - pad;

        float acc = 0.0f;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            int w_ic_base = w_oc_base + ic * 9;
            int x_ic_base = ((n * Cin + ic) * Hin) * Win;

            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                int iy = in_y0 + ky;
                if ((unsigned)iy >= (unsigned)Hin) continue;
                int x_row = x_ic_base + iy * Win;
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = in_x0 + kx;
                    if ((unsigned)ix >= (unsigned)Win) continue;
                    float xv = x[x_row + ix];
                    float wv = w[w_ic_base + ky * 3 + kx];
                    acc = fmaf(xv, wv, acc);
                }
            }
        }

        acc = bn_relu(acc, a, b);
        y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
    }
}

__global__ __launch_bounds__(256, 2)
void conv2d_k1_bn_relu_nchw_generic(
    const float* __restrict__ x,     // [N,Cin,Hin,Win]
    const float* __restrict__ w,     // [Cout,Cin]
    const float* __restrict__ alpha, // [Cout]
    const float* __restrict__ beta,  // [Cout]
    float* __restrict__ y,           // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride
) {
    int nc = (int)blockIdx.y;
    int n = nc / Cout;
    int oc = nc - n * Cout;
    if (n >= N) return;

    float a = LDG(alpha + oc);
    float b = LDG(beta + oc);

    int spatial_out = Hout * Wout;
    int base_sp = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

    int w_oc_base = oc * Cin;
    int spatial_in = Hin * Win;

    for (int sp = base_sp; sp < spatial_out; sp += (int)gridDim.x * (int)blockDim.x) {
        int oh = sp / Wout;
        int ow = sp - oh * Wout;

        int iy = oh * stride;
        int ix = ow * stride;

        float acc = 0.0f;
        int x_base = ((n * Cin) * Hin + iy) * Win + ix;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            float xv = x[x_base + ic * spatial_in];
            float wv = w[w_oc_base + ic];
            acc = fmaf(xv, wv, acc);
        }

        acc = bn_relu(acc, a, b);
        y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
    }
}

// ------------------------------------------------------------
// Specialized conv1 kernel: Cin=3, Cout=32, k=3, stride=2, pad=1
// Block computes a tile of output (TILE_H x TILE_W).
// Threads are organized as:
//  - 8x8 thread tile => 64 threads cover 8x8 output pixels; each thread computes 4 output channels.
//  - We run with 256 threads: 4 "channel groups" (0..3) mapped to warps/threads.
//  - Shared memory holds input tile for all 3 channels: (IN_H x IN_W x 3).
// ------------------------------------------------------------

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

template<int TILE_H, int TILE_W>
__global__ __launch_bounds__(256, 2)
void conv1_k3s2p1_bn_relu_nchw_tiled(
    const float* __restrict__ x,     // [N,3,H,W]
    const float* __restrict__ w,     // [32,3,3,3]
    const float* __restrict__ alpha, // [32]
    const float* __restrict__ beta,  // [32]
    float* __restrict__ y,           // [N,32,Hout,Wout]
    int N, int Hin, int Win,
    int Hout, int Wout
) {
    // blockIdx.z: batch
    int n = (int)blockIdx.z;
    if (n >= N) return;

    // blockIdx.x/y: tile in output space
    int tile_oh0 = (int)blockIdx.y * TILE_H;
    int tile_ow0 = (int)blockIdx.x * TILE_W;

    // Map threads:
    //  - tid 0..255
    //  - pixel threads: pix_tid = tid & 63  => (ph,pw) in 8x8
    //  - channel group: cg = tid >> 6      => 0..3, each group computes 8 output channels (4 per thread via lane split)
    int tid = (int)threadIdx.x;
    int pix_tid = tid & 63;      // 0..63
    int cg = tid >> 6;           // 0..3

    // 8x8 pixels
    int ph = pix_tid >> 3;       // 0..7
    int pw = pix_tid & 7;        // 0..7

    // Each thread computes 2 output channels within its group, to keep regs modest:
    //  - lane within 64-thread pixel group
    int lane = pix_tid; // 0..63
    // Two channels per thread: ch_in_group = (lane & 1) ? 1 : 0, and base offset by (lane >> 1) % 4? that's messy.
    // Simpler: each pixel has 32 channels. We have 256 threads = 4 groups * 64 pixels.
    // For each pixel, we need 32 outputs. We'll compute 8 outputs per group, and within a group, each pixel-thread computes all 8 outputs sequentially.
    // That is 8 accumulators per thread; with Cin=3 and 3x3, still manageable.
    int oc_base = cg * 8; // 0,8,16,24

    // Shared memory input tile:
    // Input needed for TILE_HxTILE_W output with stride2 and k3:
    // in_h range: [tile_oh0*2 -1, (tile_oh0 + TILE_H -1)*2 -1 +2] => length = TILE_H*2 + 2
    // similarly for width.
    constexpr int IN_H = TILE_H * 2 + 2;
    constexpr int IN_W = TILE_W * 2 + 2;

    extern __shared__ float smem[]; // size = 3*IN_H*IN_W
    float* sx0 = smem;
    float* sx1 = smem + IN_H * IN_W;
    float* sx2 = smem + 2 * IN_H * IN_W;

    // Cooperative load: 256 threads load 3*IN_H*IN_W elements.
    int total = 3 * IN_H * IN_W;
    int base_in_y = tile_oh0 * 2 - 1;
    int base_in_x = tile_ow0 * 2 - 1;

    int n_base = n * 3 * Hin * Win;

    for (int idx = tid; idx < total; idx += 256) {
        int c = idx / (IN_H * IN_W);
        int rem = idx - c * (IN_H * IN_W);
        int iy = rem / IN_W;
        int ix = rem - iy * IN_W;

        int gy = base_in_y + iy;
        int gx = base_in_x + ix;

        float v = 0.0f;
        if ((unsigned)gy < (unsigned)Hin && (unsigned)gx < (unsigned)Win) {
            int gidx = n_base + (c * Hin + gy) * Win + gx;
            v = x[gidx];
        }
        if (c == 0) sx0[iy * IN_W + ix] = v;
        else if (c == 1) sx1[iy * IN_W + ix] = v;
        else sx2[iy * IN_W + ix] = v;
    }
    __syncthreads();

    // Compute up to 1 output pixel for this thread (ph,pw within tile).
    int oh = tile_oh0 + ph;
    int ow = tile_ow0 + pw;
    if (oh >= Hout || ow >= Wout) return;

    // Input top-left in shared for this output:
    int in_y = ph * 2;
    int in_x = pw * 2;

    // Accumulate 8 output channels for this group.
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

    // Weight layout: w[oc][ic][ky][kx]
    // Unroll ic,ky,kx (small fixed sizes).
    #pragma unroll
    for (int ic = 0; ic < 3; ++ic) {
        const float* s = (ic == 0) ? sx0 : (ic == 1 ? sx1 : sx2);
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int sy = in_y + ky;
            int srow = sy * IN_W;
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                float xv = s[srow + (in_x + kx)];
                int wbase = (oc_base * 3 + ic) * 9 + ky * 3 + kx;
                float w0 = LDG(w + wbase + 0 * 27);
                float w1 = LDG(w + wbase + 1 * 27);
                float w2 = LDG(w + wbase + 2 * 27);
                float w3 = LDG(w + wbase + 3 * 27);
                float w4 = LDG(w + wbase + 4 * 27);
                float w5 = LDG(w + wbase + 5 * 27);
                float w6 = LDG(w + wbase + 6 * 27);
                float w7 = LDG(w + wbase + 7 * 27);
                acc0 = fmaf(xv, w0, acc0);
                acc1 = fmaf(xv, w1, acc1);
                acc2 = fmaf(xv, w2, acc2);
                acc3 = fmaf(xv, w3, acc3);
                acc4 = fmaf(xv, w4, acc4);
                acc5 = fmaf(xv, w5, acc5);
                acc6 = fmaf(xv, w6, acc6);
                acc7 = fmaf(xv, w7, acc7);
            }
        }
    }

    // BN + ReLU and store
    int out_base = ((n * 32) * Hout + oh) * Wout + ow; // points to oc=0
    // For each oc in group, output is at out_base + oc*Hout*Wout
    int oc_stride = Hout * Wout;

    float a0 = LDG(alpha + (oc_base + 0)), b0 = LDG(beta + (oc_base + 0));
    float a1 = LDG(alpha + (oc_base + 1)), b1 = LDG(beta + (oc_base + 1));
    float a2 = LDG(alpha + (oc_base + 2)), b2 = LDG(beta + (oc_base + 2));
    float a3 = LDG(alpha + (oc_base + 3)), b3 = LDG(beta + (oc_base + 3));
    float a4 = LDG(alpha + (oc_base + 4)), b4 = LDG(beta + (oc_base + 4));
    float a5 = LDG(alpha + (oc_base + 5)), b5 = LDG(beta + (oc_base + 5));
    float a6 = LDG(alpha + (oc_base + 6)), b6 = LDG(beta + (oc_base + 6));
    float a7 = LDG(alpha + (oc_base + 7)), b7 = LDG(beta + (oc_base + 7));

    y[out_base + (oc_base + 0) * oc_stride] = bn_relu(acc0, a0, b0);
    y[out_base + (oc_base + 1) * oc_stride] = bn_relu(acc1, a1, b1);
    y[out_base + (oc_base + 2) * oc_stride] = bn_relu(acc2, a2, b2);
    y[out_base + (oc_base + 3) * oc_stride] = bn_relu(acc3, a3, b3);
    y[out_base + (oc_base + 4) * oc_stride] = bn_relu(acc4, a4, b4);
    y[out_base + (oc_base + 5) * oc_stride] = bn_relu(acc5, a5, b5);
    y[out_base + (oc_base + 6) * oc_stride] = bn_relu(acc6, a6, b6);
    y[out_base + (oc_base + 7) * oc_stride] = bn_relu(acc7, a7, b7);
}

// ------------------------------------------------------------
// Host API
// ------------------------------------------------------------

torch::Tensor conv_bn_relu_forward_cuda(
    torch::Tensor x,            // [N,Cin,H,W]
    torch::Tensor w,            // [Cout,Cin,Kh,Kw]
    torch::Tensor alpha,        // [Cout]
    torch::Tensor beta,         // [Cout]
    int64_t stride,
    int64_t pad,
    int64_t kh,
    int64_t kw
) {
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(alpha); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(alpha); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT(x); CHECK_FLOAT(w); CHECK_FLOAT(alpha); CHECK_FLOAT(beta);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be [Cout,Cin,Kh,Kw]");
    TORCH_CHECK(alpha.dim() == 1 && beta.dim() == 1, "alpha/beta must be 1D");
    TORCH_CHECK(kh == w.size(2) && kw == w.size(3), "kh/kw must match weight");
    TORCH_CHECK((kh == 3 && kw == 3) || (kh == 1 && kw == 1), "only 3x3 or 1x1 supported");
    TORCH_CHECK(stride == 1 || stride == 2, "stride must be 1 or 2");
    if (kh == 3) TORCH_CHECK(pad == 1, "3x3 expects pad=1");
    if (kh == 1) TORCH_CHECK(pad == 0, "1x1 expects pad=0");

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);
    int Cout = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == Cin, "weight Cin mismatch");
    TORCH_CHECK((int)alpha.numel() == Cout && (int)beta.numel() == Cout, "alpha/beta must be [Cout]");

    int Hout = (int)((Hin + 2 * (int)pad - (int)kh) / (int)stride + 1);
    int Wout = (int)((Win + 2 * (int)pad - (int)kw) / (int)stride + 1);

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    // Specialized fast path for conv1 exact shape
    if (kh == 3 && kw == 3 && stride == 2 && pad == 1 && Cin == 3 && Cout == 32) {
        constexpr int TILE_H = 8;
        constexpr int TILE_W = 8;
        dim3 block(256, 1, 1);
        dim3 grid((Wout + TILE_W - 1) / TILE_W, (Hout + TILE_H - 1) / TILE_H, N);
        size_t smem_bytes = (size_t)3 * (TILE_H * 2 + 2) * (TILE_W * 2 + 2) * sizeof(float);
        conv1_k3s2p1_bn_relu_nchw_tiled<TILE_H, TILE_W><<<grid, block, smem_bytes>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Hin, Win, Hout, Wout
        );
        CUDA_KERNEL_CHECK();
        return y;
    }

    // Generic fallback
    int block = 256;
    int grid_y = N * Cout;
    int spatial = Hout * Wout;
    int grid_x = (spatial + block - 1) / block;
    if (grid_x > 1024) grid_x = 1024;
    dim3 grid(grid_x, grid_y, 1);

    if (kh == 3) {
        conv2d_k3_bn_relu_nchw_generic<<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Cin, Hin, Win, Cout, Hout, Wout,
            (int)stride, (int)pad
        );
    } else {
        conv2d_k1_bn_relu_nchw_generic<<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Cin, Hin, Win, Cout, Hout, Wout,
            (int)stride
        );
    }
    CUDA_KERNEL_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_bn_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor alpha,
    torch::Tensor beta,
    int64_t stride,
    int64_t pad,
    int64_t kh,
    int64_t kw
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_efficientnetb2_convbnrelu_v4_tiledconv1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_bn_relu_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model definition
# -------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)

        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                expanded_channels, expanded_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=expanded_channels, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())

        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

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

    def _conv_bn_relu_fused(self, x, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        # Fast path only for inference (BN uses running stats).
        if not (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and (not self.training) and (not bn.training)):
            return self.relu(bn(conv(x)))

        if bn.track_running_stats is False:
            return self.relu(bn(conv(x)))

        if not (conv.weight.is_cuda and bn.weight.is_cuda and bn.running_mean.is_cuda and bn.running_var.is_cuda):
            return self.relu(bn(conv(x)))

        w = conv.weight.contiguous()
        alpha, beta = self._bn_alpha_beta_infer(bn, x.device, x.dtype)

        kh = int(w.size(2))
        kw = int(w.size(3))
        stride = int(conv.stride[0])
        pad = int(conv.padding[0])

        if not ((kh == 3 and kw == 3 and pad == 1) or (kh == 1 and kw == 1 and pad == 0)):
            return self.relu(bn(conv(x)))

        return custom_ops_lib.conv_bn_relu_forward_cuda(x, w, alpha, beta, stride, pad, kh, kw)

    def forward(self, x):
        x = x.contiguous()
        x = self._conv_bn_relu_fused(x, self.conv1, self.bn1)

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)

        x = x.contiguous()
        x = self._conv_bn_relu_fused(x, self.conv_final, self.bn_final)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x