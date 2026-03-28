import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension
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

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// -------------------------
// Constant-memory specialization for conv1 weights when Cin=3,Cout=64,k=3
// Layout matches OIHW contiguous: [oc][ic][ky][kx], total 64*3*3*3 = 1728? Wait: 64*3*9=1728.
// Store 1728 floats.
__constant__ float c_w1_64x3x3x3[64 * 3 * 3 * 3];

static inline void load_w1_to_constant_if_needed(const float* w_host_or_device, size_t bytes) {
    // Caller ensures bytes <= sizeof(c_w1_64x3x3x3)
    cudaMemcpyToSymbol(c_w1_64x3x3x3, w_host_or_device, bytes, 0, cudaMemcpyDeviceToDevice);
}

// -------------------------
// Baseline direct conv kernels (general)
// -------------------------

__global__ void conv2d_fwd_nchw_k3(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin,3,3]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride, int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int in_y0 = oh * stride - pad;
    int in_x0 = ow * stride - pad;

    float acc = 0.0f;
    int w_oc_base = oc * Cin * 9;

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

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

__global__ void conv2d_fwd_nchw_k1(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int iy = oh * stride;
    int ix = ow * stride;

    float acc = 0.0f;
    int w_oc_base = oc * Cin;
    int x_base = ((n * Cin) * Hin + iy) * Win + ix;

    for (int ic = 0; ic < Cin; ++ic) {
        float xv = x[x_base + ic * Hin * Win];
        float wv = w[w_oc_base + ic];
        acc = fmaf(xv, wv, acc);
    }

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

__global__ void bn_inplace(
    float* __restrict__ x,               // [N,C,H,W]
    const float* __restrict__ alpha,     // [C]
    const float* __restrict__ beta,      // [C]
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int t1 = idx / W;
    int h = t1 % H;
    int t2 = t1 / H;
    int c = t2 % C;
    int n = t2 / C;
    int off = ((n * C + c) * H + h) * W + w;

    float v = x[off];
    v = fmaf(v, ldg_f32(alpha + c), ldg_f32(beta + c));
    x[off] = v;
}

__global__ void bn_relu_inplace(
    float* __restrict__ x,               // [N,C,H,W]
    const float* __restrict__ alpha,     // [C]
    const float* __restrict__ beta,      // [C]
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int t1 = idx / W;
    int h = t1 % H;
    int t2 = t1 / H;
    int c = t2 % C;
    int n = t2 / C;
    int off = ((n * C + c) * H + h) * W + w;

    float v = x[off];
    v = fmaf(v, ldg_f32(alpha + c), ldg_f32(beta + c));
    v = v > 0.0f ? v : 0.0f;
    x[off] = v;
}

__global__ void bn_add_relu(
    const float* __restrict__ x,         // main path [N,C,H,W]
    const float* __restrict__ res,       // residual [N,C,H,W]
    const float* __restrict__ alpha,     // [C]
    const float* __restrict__ beta,      // [C]
    float* __restrict__ y,               // [N,C,H,W]
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int t1 = idx / W;
    int h = t1 % H;
    int t2 = t1 / H;
    int c = t2 % C;
    int n = t2 / C;
    int off = ((n * C + c) * H + h) * W + w;

    float v = x[off];
    v = fmaf(v, ldg_f32(alpha + c), ldg_f32(beta + c)); // BN
    v += res[off];                                      // Add residual
    v = v > 0.0f ? v : 0.0f;                            // ReLU
    y[off] = v;
}

// -------------------------
// Specialized conv1 kernel: Cin=3, Cout=64, k=3, pad=1, stride=1|2, BN+ReLU fused
// Uses 3D grid: x=ow tiles, y=oh tiles, z=(n*Cout + oc) blocks
// Each block computes a small tile of ow for one (n,oc,oh).
// This removes heavy div/mod and uses constant memory for weights.
// -------------------------

__global__ __launch_bounds__(128, 2) void conv1_3x3_cin3_cout64_bn_relu(
    const float* __restrict__ x,       // [N,3,H,W]
    const float* __restrict__ alpha,   // [64]
    const float* __restrict__ beta,    // [64]
    float* __restrict__ y,             // [N,64,Hout,Wout]
    int N, int Hin, int Win,
    int Hout, int Wout,
    int stride, int pad
) {
    int ow = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int oh = (int)blockIdx.y;
    int noz = (int)blockIdx.z; // n*Cout + oc
    int oc = noz & 63;         // Cout=64
    int n  = noz >> 6;

    if (n >= N || ow >= Wout) return;

    int in_y0 = oh * stride - pad;
    int in_x0 = ow * stride - pad;

    float acc = 0.0f;
    // weights base in constant memory
    int w_oc_base = oc * 3 * 9; // Cin=3
    #pragma unroll
    for (int ic = 0; ic < 3; ++ic) {
        int x_ic_base = ((n * 3 + ic) * Hin) * Win;
        int w_ic_base = w_oc_base + ic * 9;
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
                float wv = c_w1_64x3x3x3[w_ic_base + ky * 3 + kx];
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    float v = fmaf(acc, ldg_f32(alpha + oc), ldg_f32(beta + oc));
    v = v > 0.0f ? v : 0.0f;
    y[(((n * 64 + oc) * Hout + oh) * Wout) + ow] = v;
}

// -------------------------
// Launch helpers
// -------------------------

static inline void launch_conv3_general(
    torch::Tensor x, torch::Tensor w, torch::Tensor y,
    int stride, int pad
) {
    int N = (int)x.size(0), Cin = (int)x.size(1), Hin = (int)x.size(2), Win = (int)x.size(3);
    int Cout = (int)w.size(0);
    int Hout = (int)y.size(2), Wout = (int)y.size(3);
    int total = N * Cout * Hout * Wout;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_fwd_nchw_k3<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout, stride, pad
    );
}

static inline void launch_conv1_general(
    torch::Tensor x, torch::Tensor w_flat, torch::Tensor y,
    int stride
) {
    int N = (int)x.size(0), Cin = (int)x.size(1), Hin = (int)x.size(2), Win = (int)x.size(3);
    int Cout = (int)w_flat.size(0);
    int Hout = (int)y.size(2), Wout = (int)y.size(3);
    int total = N * Cout * Hout * Wout;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_fwd_nchw_k1<<<grid, block>>>(
        x.data_ptr<float>(), w_flat.data_ptr<float>(), y.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout, stride
    );
}

static inline void launch_conv1_specialized_64x3(
    torch::Tensor x, torch::Tensor w1, torch::Tensor a1, torch::Tensor b1, torch::Tensor y,
    int stride, int pad
) {
    // Copy weights to constant memory (device-to-device) once per call (cheap: 1728 floats).
    load_w1_to_constant_if_needed(w1.data_ptr<float>(), sizeof(float) * 64 * 3 * 3 * 3);

    int N = (int)x.size(0);
    int Hin = (int)x.size(2), Win = (int)x.size(3);
    int Hout = (int)y.size(2), Wout = (int)y.size(3);

    dim3 block(128, 1, 1);
    dim3 grid((Wout + block.x - 1) / block.x, (unsigned)Hout, (unsigned)(N * 64));
    conv1_3x3_cin3_cout64_bn_relu<<<grid, block>>>(
        x.data_ptr<float>(),
        a1.data_ptr<float>(), b1.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Hin, Win, Hout, Wout,
        stride, pad
    );
}

torch::Tensor resnet_basic_block_forward_cuda(
    torch::Tensor x,            // [N,Cin,H,W]
    torch::Tensor w1,           // [Cout,Cin,3,3]
    torch::Tensor a1,           // [Cout]
    torch::Tensor b1,           // [Cout]
    torch::Tensor w2,           // [Cout,Cout,3,3]
    torch::Tensor a2,           // [Cout]
    torch::Tensor b2,           // [Cout]
    torch::Tensor wds,          // [Cout,Cin,1,1]
    torch::Tensor ads,          // [Cout]
    torch::Tensor bds,          // [Cout]
    int64_t stride1
) {
    CHECK_CUDA(x); CHECK_CUDA(w1); CHECK_CUDA(w2); CHECK_CUDA(wds);
    CHECK_CUDA(a1); CHECK_CUDA(b1); CHECK_CUDA(a2); CHECK_CUDA(b2);
    CHECK_CUDA(ads); CHECK_CUDA(bds);

    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w1); CHECK_CONTIGUOUS(w2); CHECK_CONTIGUOUS(wds);
    CHECK_CONTIGUOUS(a1); CHECK_CONTIGUOUS(b1); CHECK_CONTIGUOUS(a2); CHECK_CONTIGUOUS(b2);
    CHECK_CONTIGUOUS(ads); CHECK_CONTIGUOUS(bds);

    CHECK_FLOAT(x); CHECK_FLOAT(w1); CHECK_FLOAT(w2); CHECK_FLOAT(wds);
    CHECK_FLOAT(a1); CHECK_FLOAT(b1); CHECK_FLOAT(a2); CHECK_FLOAT(b2);
    CHECK_FLOAT(ads); CHECK_FLOAT(bds);

    TORCH_CHECK(x.dim()==4, "x must be NCHW");
    TORCH_CHECK(w1.dim()==4 && w1.size(2)==3 && w1.size(3)==3, "w1 must be 3x3");
    TORCH_CHECK(w2.dim()==4 && w2.size(2)==3 && w2.size(3)==3, "w2 must be 3x3");
    TORCH_CHECK(wds.dim()==4 && wds.size(2)==1 && wds.size(3)==1, "wds must be 1x1");
    TORCH_CHECK(stride1==1 || stride1==2, "stride1 must be 1 or 2");

    int64_t N = x.size(0), Cin = x.size(1), Hin = x.size(2), Win = x.size(3);
    int64_t Cout = w1.size(0);
    TORCH_CHECK(w1.size(1)==Cin, "w1 Cin mismatch");
    TORCH_CHECK(w2.size(0)==Cout && w2.size(1)==Cout, "w2 shape mismatch");
    TORCH_CHECK(wds.size(0)==Cout && wds.size(1)==Cin, "wds shape mismatch");

    TORCH_CHECK(a1.numel()==Cout && b1.numel()==Cout, "bn1 params mismatch");
    TORCH_CHECK(a2.numel()==Cout && b2.numel()==Cout, "bn2 params mismatch");
    TORCH_CHECK(ads.numel()==Cout && bds.numel()==Cout, "bn_ds params mismatch");

    int64_t H1 = (Hin + 2 - 3) / stride1 + 1;
    int64_t W1 = (Win + 2 - 3) / stride1 + 1;

    // main path conv1 -> o1
    auto o1 = torch::empty({N, Cout, H1, W1}, x.options());

    // Specialized fast path only for Cin=3,Cout=64 (common for provided model)
    if (Cin == 3 && Cout == 64) {
        launch_conv1_specialized_64x3(x, w1, a1, b1, o1, (int)stride1, 1);
    } else {
        launch_conv3_general(x, w1, o1, (int)stride1, 1);
        int total = (int)(N * Cout * H1 * W1);
        int block = 256;
        int grid = (total + block - 1) / block;
        bn_relu_inplace<<<grid, block>>>(
            o1.data_ptr<float>(), a1.data_ptr<float>(), b1.data_ptr<float>(),
            (int)N, (int)Cout, (int)H1, (int)W1
        );
    }

    // conv2 -> o2
    auto o2 = torch::empty({N, Cout, H1, W1}, x.options());
    launch_conv3_general(o1, w2, o2, 1, 1);

    // residual: 1x1 conv + BN
    auto res = torch::empty({N, Cout, H1, W1}, x.options());
    {
        auto wds_flat = wds.view({Cout, Cin, 1, 1}).contiguous().view({Cout, Cin});
        launch_conv1_general(x, wds_flat, res, (int)stride1);

        int total = (int)(N * Cout * H1 * W1);
        int block = 256;
        int grid = (total + block - 1) / block;
        bn_inplace<<<grid, block>>>(
            res.data_ptr<float>(), ads.data_ptr<float>(), bds.data_ptr<float>(),
            (int)N, (int)Cout, (int)H1, (int)W1
        );
    }

    // BN2 + Add + ReLU
    auto y = torch::empty({N, Cout, H1, W1}, x.options());
    {
        int total = (int)(N * Cout * H1 * W1);
        int block = 256;
        int grid = (total + block - 1) / block;
        bn_add_relu<<<grid, block>>>(
            o2.data_ptr<float>(), res.data_ptr<float>(),
            a2.data_ptr<float>(), b2.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)Cout, (int)H1, (int)W1
        );
    }
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor resnet_basic_block_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor a2, torch::Tensor b2,
    torch::Tensor wds, torch::Tensor ads, torch::Tensor bds,
    int64_t stride1
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_res_net_basic_block_v3_constw",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["resnet_basic_block_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model definition (uses fused basic block op)
# -------------------------

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = int(stride)

    @staticmethod
    def _bn_alpha_beta(bn: nn.BatchNorm2d, device, dtype):
        gamma = bn.weight.to(device=device, dtype=dtype)
        beta = bn.bias.to(device=device, dtype=dtype)
        mean = bn.running_mean.to(device=device, dtype=dtype)
        var = bn.running_var.to(device=device, dtype=dtype)
        invstd = torch.rsqrt(var + bn.eps)
        alpha = gamma * invstd
        beta2 = beta - mean * alpha
        return alpha.contiguous(), beta2.contiguous()

    def forward(self, x):
        if (
            x.is_cuda and x.dtype == torch.float32 and (not self.training) and
            self.conv1.weight.is_cuda and self.conv2.weight.is_cuda
        ):
            x = x.contiguous()
            w1 = self.conv1.weight.contiguous()
            w2 = self.conv2.weight.contiguous()
            a1, b1 = self._bn_alpha_beta(self.bn1, x.device, x.dtype)
            a2, b2 = self._bn_alpha_beta(self.bn2, x.device, x.dtype)

            ds_conv = self.downsample[0]
            ds_bn = self.downsample[1]
            wds = ds_conv.weight.contiguous()
            ads, bds = self._bn_alpha_beta(ds_bn, x.device, x.dtype)

            return custom_ops_lib.resnet_basic_block_forward_cuda(
                x, w1, a1, b1, w2, a2, b2, wds, ads, bds, int(self.stride)
            )

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out