import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension (more fused Bottleneck for inference)
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

// Small helpers
__device__ __forceinline__ float relu(float v) { return v > 0.0f ? v : 0.0f; }

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

// ------------------------------------
// Fused kernels
// ------------------------------------

// conv1 1x1 + bn + relu : y = relu(x*w + beta + alpha*acc)
__global__ __launch_bounds__(256, 2)
void conv1x1_bn_relu_nchw(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin]
    const float* __restrict__ alpha, // [Cout]
    const float* __restrict__ beta,  // [Cout]
    float* __restrict__ y,         // [N,Cout,H,W]
    int N, int Cin, int H, int W, int Cout
) {
    int64_t HW = (int64_t)H * W;
    int64_t total = (int64_t)N * Cout * HW;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = idx;
        int ow = (int)(t % W); t /= W;
        int oh = (int)(t % H); t /= H;
        int oc = (int)(t % Cout);
        int n  = (int)(t / Cout);

        int64_t x_base = ((int64_t)n * Cin * H + oh) * W + ow; // ic=0
        int64_t w_base = (int64_t)oc * Cin;

        float acc = 0.0f;
        #pragma unroll 4
        for (int ic = 0; ic < Cin; ++ic) {
            float xv = x[x_base + (int64_t)ic * HW];
            float wv = w[w_base + ic];
            acc = fmaf(xv, wv, acc);
        }
        float a = ldg_f32(alpha + oc);
        float b = ldg_f32(beta + oc);
        float v = fmaf(acc, a, b);
        y[(((int64_t)n * Cout + oc) * H + oh) * W + ow] = relu(v);
    }
}

// downsample conv1x1 (stride=1 or 2) + bn : res = acc*alpha + beta
__global__ __launch_bounds__(256, 2)
void conv1x1_bn_stride_nchw(
    const float* __restrict__ x,   // [N,Cin,Hin,Win]
    const float* __restrict__ w,   // [Cout,Cin]
    const float* __restrict__ alpha, // [Cout]
    const float* __restrict__ beta,  // [Cout]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win, int Cout, int Hout, int Wout, int stride
) {
    int64_t HWout = (int64_t)Hout * Wout;
    int64_t total = (int64_t)N * Cout * HWout;
    int64_t HW = (int64_t)Hin * Win;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = idx;
        int ow = (int)(t % Wout); t /= Wout;
        int oh = (int)(t % Hout); t /= Hout;
        int oc = (int)(t % Cout);
        int n  = (int)(t / Cout);

        int iy = oh * stride;
        int ix = ow * stride;

        int64_t x_base = ((int64_t)n * Cin * Hin + iy) * Win + ix; // ic=0
        int64_t w_base = (int64_t)oc * Cin;

        float acc = 0.0f;
        #pragma unroll 4
        for (int ic = 0; ic < Cin; ++ic) {
            float xv = x[x_base + (int64_t)ic * HW];
            float wv = w[w_base + ic];
            acc = fmaf(xv, wv, acc);
        }
        float a = ldg_f32(alpha + oc);
        float b = ldg_f32(beta + oc);
        float v = fmaf(acc, a, b);
        y[(((int64_t)n * Cout + oc) * Hout + oh) * Wout + ow] = v;
    }
}

// conv2 3x3 (stride=1 or 2, pad=1) : plain conv
__global__ __launch_bounds__(256, 2)
void conv3x3_nchw(
    const float* __restrict__ x,   // [N,Cin,Hin,Win]
    const float* __restrict__ w,   // [Cout,Cin,3,3] contiguous
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride, int pad
) {
    int64_t total = (int64_t)N * Cout * Hout * Wout;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = idx;
        int ow = (int)(t % Wout); t /= Wout;
        int oh = (int)(t % Hout); t /= Hout;
        int oc = (int)(t % Cout);
        int n  = (int)(t / Cout);

        int in_y0 = oh * stride - pad;
        int in_x0 = ow * stride - pad;

        float acc = 0.0f;
        int64_t w_oc_base = (int64_t)oc * Cin * 9;

        for (int ic = 0; ic < Cin; ++ic) {
            int64_t w_ic_base = w_oc_base + (int64_t)ic * 9;
            int64_t x_ic_base = ((int64_t)n * Cin + ic) * Hin * Win;
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                int iy = in_y0 + ky;
                if ((unsigned)iy >= (unsigned)Hin) continue;
                int64_t x_row = x_ic_base + (int64_t)iy * Win;
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

        y[(((int64_t)n * Cout + oc) * Hout + oh) * Wout + ow] = acc;
    }
}

// bn + relu in-place (vectorized on W when possible)
__global__ __launch_bounds__(256, 2)
void bn_relu_inplace_vec(
    float* __restrict__ x,               // [N,C,H,W]
    const float* __restrict__ alpha,     // [C]
    const float* __restrict__ beta,      // [C]
    int N, int C, int H, int W
) {
    int64_t total = (int64_t)N * C * H * W;

    // Try float4 along contiguous dimension (W) if aligned and W%4==0.
    bool vec_ok = ((W & 3) == 0) && ((((uintptr_t)x) & 15) == 0);
    if (!vec_ok) {
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += (int64_t)blockDim.x * gridDim.x) {

            int64_t t = idx;
            int w = (int)(t % W); t /= W;
            int h = (int)(t % H); t /= H;
            int c = (int)(t % C);
            int n = (int)(t / C);

            int64_t off = (((int64_t)n * C + c) * H + h) * W + w;
            float a = ldg_f32(alpha + c);
            float b = ldg_f32(beta + c);
            float v = fmaf(x[off], a, b);
            x[off] = relu(v);
        }
        return;
    }

    int W4 = W >> 2;
    int64_t total4 = (int64_t)N * C * H * W4;
    float4* x4 = reinterpret_cast<float4*>(x);

    for (int64_t idx4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx4 < total4;
         idx4 += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = idx4;
        int w4 = (int)(t % W4); t /= W4;
        int h  = (int)(t % H);  t /= H;
        int c  = (int)(t % C);
        int n  = (int)(t / C);

        int64_t off4 = (((int64_t)n * C + c) * H + h) * W4 + w4;
        float a = ldg_f32(alpha + c);
        float b = ldg_f32(beta + c);

        float4 v = x4[off4];
        v.x = relu(fmaf(v.x, a, b));
        v.y = relu(fmaf(v.y, a, b));
        v.z = relu(fmaf(v.z, a, b));
        v.w = relu(fmaf(v.w, a, b));
        x4[off4] = v;
    }
}

// conv3 1x1 + bn3 + add residual + relu : y = relu((acc*alpha+beta)+res)
__global__ __launch_bounds__(256, 2)
void conv1x1_bn_add_relu_nchw(
    const float* __restrict__ x,     // [N,Cin,H,W] (input to conv3, i.e., o2)
    const float* __restrict__ w,     // [Cout,Cin]
    const float* __restrict__ res,   // [N,Cout,H,W]
    const float* __restrict__ alpha, // [Cout]
    const float* __restrict__ beta,  // [Cout]
    float* __restrict__ y,           // [N,Cout,H,W]
    int N, int Cin, int H, int W, int Cout
) {
    int64_t HW = (int64_t)H * W;
    int64_t total = (int64_t)N * Cout * HW;

    bool vec_ok = ((W & 3) == 0) &&
                  ((((uintptr_t)res) & 15) == 0) &&
                  ((((uintptr_t)y) & 15) == 0);

    if (!vec_ok) {
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += (int64_t)blockDim.x * gridDim.x) {

            int64_t t = idx;
            int ow = (int)(t % W); t /= W;
            int oh = (int)(t % H); t /= H;
            int oc = (int)(t % Cout);
            int n  = (int)(t / Cout);

            int64_t x_base = ((int64_t)n * Cin * H + oh) * W + ow;
            int64_t w_base = (int64_t)oc * Cin;

            float acc = 0.0f;
            #pragma unroll 4
            for (int ic = 0; ic < Cin; ++ic) {
                float xv = x[x_base + (int64_t)ic * HW];
                float wv = w[w_base + ic];
                acc = fmaf(xv, wv, acc);
            }
            float a = ldg_f32(alpha + oc);
            float b = ldg_f32(beta + oc);
            float v = fmaf(acc, a, b);
            int64_t off = (((int64_t)n * Cout + oc) * H + oh) * W + ow;
            v += res[off];
            y[off] = relu(v);
        }
        return;
    }

    int W4 = W >> 2;
    int64_t total4 = (int64_t)N * Cout * H * W4;
    const float4* res4 = reinterpret_cast<const float4*>(res);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int64_t idx4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx4 < total4;
         idx4 += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = idx4;
        int w4 = (int)(t % W4); t /= W4;
        int oh = (int)(t % H);  t /= H;
        int oc = (int)(t % Cout);
        int n  = (int)(t / Cout);

        float a = ldg_f32(alpha + oc);
        float b = ldg_f32(beta + oc);

        int ow0 = w4 * 4;

        // compute 4 outputs; weights reused, x loads different ow
        float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f;
        int64_t w_base = (int64_t)oc * Cin;
        int64_t x_base0 = ((int64_t)n * Cin * H + oh) * W + ow0;

        #pragma unroll 4
        for (int ic = 0; ic < Cin; ++ic) {
            float wv = w[w_base + ic];
            int64_t xptr = x_base0 + (int64_t)ic * HW;
            float xv0 = x[xptr + 0];
            float xv1 = x[xptr + 1];
            float xv2 = x[xptr + 2];
            float xv3 = x[xptr + 3];
            acc0 = fmaf(xv0, wv, acc0);
            acc1 = fmaf(xv1, wv, acc1);
            acc2 = fmaf(xv2, wv, acc2);
            acc3 = fmaf(xv3, wv, acc3);
        }

        float4 r = res4[(((int64_t)n * Cout + oc) * H + oh) * W4 + w4];
        float4 out;
        out.x = relu(fmaf(acc0, a, b) + r.x);
        out.y = relu(fmaf(acc1, a, b) + r.y);
        out.z = relu(fmaf(acc2, a, b) + r.z);
        out.w = relu(fmaf(acc3, a, b) + r.w);

        y4[(((int64_t)n * Cout + oc) * H + oh) * W4 + w4] = out;
    }
}

static inline int clamp_grid(int grid) {
    // avoid too-large grids on small tensors; allow enough CTAs to fill GPU
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    return grid;
}

torch::Tensor bottleneck_forward_cuda(
    torch::Tensor x,               // [N,Cin,H,W]
    // conv1 1x1 -> bn1 -> relu (fused)
    torch::Tensor w1,              // [Cmid,Cin,1,1]
    torch::Tensor a1, torch::Tensor b1, // [Cmid]
    // conv2 3x3 -> bn2 -> relu (bn+relu fused kernel)
    torch::Tensor w2,              // [Cmid,Cmid,3,3]
    torch::Tensor a2, torch::Tensor b2, // [Cmid]
    // conv3 1x1 -> bn3 + add + relu (fused)
    torch::Tensor w3,              // [Cout,Cmid,1,1]
    torch::Tensor a3, torch::Tensor b3, // [Cout]
    // downsample (optional): conv1x1 -> bn (fused)
    torch::Tensor w_ds,            // [] or [Cout,Cin,1,1]
    torch::Tensor a_ds, torch::Tensor b_ds, // [] or [Cout]
    int64_t stride2,               // 1 or 2 (for conv2 and downsample conv)
    bool has_downsample
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x);
    CHECK_CUDA(w1); CHECK_CONTIGUOUS(w1); CHECK_FLOAT(w1);
    CHECK_CUDA(w2); CHECK_CONTIGUOUS(w2); CHECK_FLOAT(w2);
    CHECK_CUDA(w3); CHECK_CONTIGUOUS(w3); CHECK_FLOAT(w3);
    CHECK_CUDA(a1); CHECK_CONTIGUOUS(a1); CHECK_FLOAT(a1);
    CHECK_CUDA(b1); CHECK_CONTIGUOUS(b1); CHECK_FLOAT(b1);
    CHECK_CUDA(a2); CHECK_CONTIGUOUS(a2); CHECK_FLOAT(a2);
    CHECK_CUDA(b2); CHECK_CONTIGUOUS(b2); CHECK_FLOAT(b2);
    CHECK_CUDA(a3); CHECK_CONTIGUOUS(a3); CHECK_FLOAT(a3);
    CHECK_CUDA(b3); CHECK_CONTIGUOUS(b3); CHECK_FLOAT(b3);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w1.dim() == 4 && w1.size(2) == 1 && w1.size(3) == 1, "w1 must be 1x1");
    TORCH_CHECK(w2.dim() == 4 && w2.size(2) == 3 && w2.size(3) == 3, "w2 must be 3x3");
    TORCH_CHECK(w3.dim() == 4 && w3.size(2) == 1 && w3.size(3) == 1, "w3 must be 1x1");
    TORCH_CHECK(stride2 == 1 || stride2 == 2, "stride2 must be 1 or 2");

    int N = (int)x.size(0), Cin = (int)x.size(1), Hin = (int)x.size(2), Win = (int)x.size(3);
    int Cmid = (int)w1.size(0);
    TORCH_CHECK((int)w1.size(1) == Cin, "w1 Cin mismatch");
    TORCH_CHECK((int)w2.size(0) == Cmid && (int)w2.size(1) == Cmid, "w2 shape mismatch");
    TORCH_CHECK((int)w3.size(1) == Cmid, "w3 Cmid mismatch");
    int Cout = (int)w3.size(0);

    TORCH_CHECK((int)a1.numel() == Cmid && (int)b1.numel() == Cmid, "bn1 params mismatch");
    TORCH_CHECK((int)a2.numel() == Cmid && (int)b2.numel() == Cmid, "bn2 params mismatch");
    TORCH_CHECK((int)a3.numel() == Cout && (int)b3.numel() == Cout, "bn3 params mismatch");

    // Views of weights as [Cout, Cin] contiguous
    auto w1_2d = w1.view({Cmid, Cin});
    auto w3_2d = w3.view({Cout, Cmid});

    // conv1+bn+relu
    auto o1 = torch::empty({N, Cmid, Hin, Win}, x.options());
    {
        int64_t total = (int64_t)N * Cmid * Hin * Win;
        int block = 256;
        int grid = clamp_grid((int)((total + block - 1) / block));
        // cap grid a bit to avoid too many tiny CTAs
        grid = min(grid, 4096);
        conv1x1_bn_relu_nchw<<<grid, block>>>(
            x.data_ptr<float>(),
            w1_2d.data_ptr<float>(),
            a1.data_ptr<float>(),
            b1.data_ptr<float>(),
            o1.data_ptr<float>(),
            N, Cin, Hin, Win, Cmid
        );
    }

    // conv2 3x3
    int H2 = (Hin + 2 - 3) / (int)stride2 + 1;
    int W2 = (Win + 2 - 3) / (int)stride2 + 1;
    auto o2 = torch::empty({N, Cmid, H2, W2}, x.options());
    {
        int64_t total = (int64_t)N * Cmid * H2 * W2;
        int block = 256;
        int grid = clamp_grid((int)((total + block - 1) / block));
        grid = min(grid, 4096);
        conv3x3_nchw<<<grid, block>>>(
            o1.data_ptr<float>(),
            w2.data_ptr<float>(),
            o2.data_ptr<float>(),
            N, Cmid, Hin, Win,
            Cmid, H2, W2,
            (int)stride2, 1
        );
    }

    // bn2+relu inplace (vectorized)
    {
        int64_t total = (int64_t)N * Cmid * H2 * W2;
        int block = 256;
        int grid = clamp_grid((int)((total + block - 1) / block));
        grid = min(grid, 4096);
        bn_relu_inplace_vec<<<grid, block>>>(
            o2.data_ptr<float>(),
            a2.data_ptr<float>(),
            b2.data_ptr<float>(),
            N, Cmid, H2, W2
        );
    }

    // residual path
    torch::Tensor res;
    if (has_downsample) {
        CHECK_CUDA(w_ds); CHECK_CONTIGUOUS(w_ds); CHECK_FLOAT(w_ds);
        CHECK_CUDA(a_ds); CHECK_CONTIGUOUS(a_ds); CHECK_FLOAT(a_ds);
        CHECK_CUDA(b_ds); CHECK_CONTIGUOUS(b_ds); CHECK_FLOAT(b_ds);
        TORCH_CHECK(w_ds.dim() == 4 && w_ds.size(2) == 1 && w_ds.size(3) == 1, "w_ds must be 1x1");
        TORCH_CHECK((int)w_ds.size(0) == Cout && (int)w_ds.size(1) == Cin, "w_ds shape mismatch");
        TORCH_CHECK((int)a_ds.numel() == Cout && (int)b_ds.numel() == Cout, "bn_ds params mismatch");

        auto wds_2d = w_ds.view({Cout, Cin});
        res = torch::empty({N, Cout, H2, W2}, x.options());
        int64_t total = (int64_t)N * Cout * H2 * W2;
        int block = 256;
        int grid = clamp_grid((int)((total + block - 1) / block));
        grid = min(grid, 4096);
        conv1x1_bn_stride_nchw<<<grid, block>>>(
            x.data_ptr<float>(),
            wds_2d.data_ptr<float>(),
            a_ds.data_ptr<float>(),
            b_ds.data_ptr<float>(),
            res.data_ptr<float>(),
            N, Cin, Hin, Win, Cout, H2, W2, (int)stride2
        );
    } else {
        TORCH_CHECK(Cin == Cout && Hin == H2 && Win == W2, "identity shape mismatch without downsample");
        res = x;
    }

    // conv3 + bn3 + add + relu (fused)
    auto y = torch::empty({N, Cout, H2, W2}, x.options());
    {
        int64_t total = (int64_t)N * Cout * H2 * W2;
        int block = 256;
        int grid = clamp_grid((int)((total + block - 1) / block));
        grid = min(grid, 4096);
        conv1x1_bn_add_relu_nchw<<<grid, block>>>(
            o2.data_ptr<float>(),
            w3_2d.data_ptr<float>(),
            res.data_ptr<float>(),
            a3.data_ptr<float>(),
            b3.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Cmid, H2, W2, Cout
        );
    }
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor bottleneck_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor a2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor a3, torch::Tensor b3,
    torch::Tensor w_ds, torch::Tensor a_ds, torch::Tensor b_ds,
    int64_t stride2, bool has_downsample
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_resnet101_bottleneck_fused_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bottleneck_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model definition (uses fused bottleneck op in eval)
# -------------------------

class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self._cached = {}  # (device, dtype) -> packed params

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

    def _get_packed(self, device, dtype):
        key = (device, dtype)
        packed = self._cached.get(key, None)
        if packed is not None:
            return packed

        w1 = self.conv1.weight.to(device=device, dtype=dtype).contiguous()
        w2 = self.conv2.weight.to(device=device, dtype=dtype).contiguous()
        w3 = self.conv3.weight.to(device=device, dtype=dtype).contiguous()
        a1, b1 = self._bn_alpha_beta(self.bn1, device, dtype)
        a2, b2 = self._bn_alpha_beta(self.bn2, device, dtype)
        a3, b3 = self._bn_alpha_beta(self.bn3, device, dtype)

        has_ds = self.downsample is not None
        if has_ds:
            ds_conv = self.downsample[0]
            ds_bn = self.downsample[1]
            wds = ds_conv.weight.to(device=device, dtype=dtype).contiguous()
            ads, bds = self._bn_alpha_beta(ds_bn, device, dtype)
        else:
            wds = torch.empty(0, device=device, dtype=dtype)
            ads = torch.empty(0, device=device, dtype=dtype)
            bds = torch.empty(0, device=device, dtype=dtype)

        packed = (w1, a1, b1, w2, a2, b2, w3, a3, b3, wds, ads, bds, bool(has_ds))
        self._cached[key] = packed
        return packed

    def forward(self, x):
        if (
            x.is_cuda and x.dtype == torch.float32 and (not self.training) and
            self.conv1.weight.is_cuda and self.conv2.weight.is_cuda and self.conv3.weight.is_cuda
        ):
            x = x.contiguous()
            w1, a1, b1, w2, a2, b2, w3, a3, b3, wds, ads, bds, has_ds = self._get_packed(x.device, x.dtype)
            return custom_ops_lib.bottleneck_forward_cuda(
                x, w1, a1, b1, w2, a2, b2, w3, a3, b3,
                wds, ads, bds,
                int(self.stride), bool(has_ds)
            )

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x