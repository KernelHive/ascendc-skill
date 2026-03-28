import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv3d_maxpool_logsumexp_relu (forward only) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// One thread computes one output element:
// y[n, 0, pd, ph, pw] where y = relu(logsumexp_c(maxpool2(conv3d(x,w,b))))
__global__ void conv3d_maxpool_logsumexp_relu_fwd_kernel(
    const float* __restrict__ x,   // [N, Cin, D, H, W]
    const float* __restrict__ w,   // [Cout, Cin, Kd, Kh, Kw]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N, 1, Dp, Hp, Wp]
    int N, int Cin, int D, int H, int W,
    int Cout, int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int Dc, int Hc, int Wc,     // conv output sizes
    int Dp, int Hp, int Wp      // pooled output sizes (kernel=2,stride=2,ceil_mode=False)
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Dp * Hp * Wp;
    if (idx >= total) return;

    int pw = idx % Wp; idx /= Wp;
    int ph = idx % Hp; idx /= Hp;
    int pd = idx % Dp; idx /= Dp;
    int n  = idx;

    // pooled window origin in conv-output coordinates
    int od0 = pd * 2;
    int oh0 = ph * 2;
    int ow0 = pw * 2;

    // First pass: compute per-channel pooled max and track global max for stable logsumexp
    float global_max = -INFINITY;

    for (int co = 0; co < Cout; ++co) {
        float m = -INFINITY;

        // maxpool kernel=2,stride=2 over conv output (Dc,Hc,Wc)
        #pragma unroll
        for (int tz = 0; tz < 2; ++tz) {
            int od = od0 + tz;
            if ((unsigned)od >= (unsigned)Dc) continue;
            #pragma unroll
            for (int ty = 0; ty < 2; ++ty) {
                int oh = oh0 + ty;
                if ((unsigned)oh >= (unsigned)Hc) continue;
                #pragma unroll
                for (int tx = 0; tx < 2; ++tx) {
                    int ow_ = ow0 + tx;
                    if ((unsigned)ow_ >= (unsigned)Wc) continue;

                    // Compute conv output at (n,co,od,oh,ow_)
                    int in_d0 = od * stride_d - pad_d;
                    int in_h0 = oh * stride_h - pad_h;
                    int in_w0 = ow_ * stride_w - pad_w;

                    float acc = (b != nullptr) ? b[co] : 0.0f;

                    for (int ci = 0; ci < Cin; ++ci) {
                        int w_base = ((co * Cin + ci) * Kd * Kh * Kw);
                        int x_base = ((n * Cin + ci) * D * H * W);

                        for (int kd = 0; kd < Kd; ++kd) {
                            int id = in_d0 + kd;
                            if ((unsigned)id >= (unsigned)D) continue;
                            for (int kh = 0; kh < Kh; ++kh) {
                                int ih = in_h0 + kh;
                                if ((unsigned)ih >= (unsigned)H) continue;

                                int x_row = x_base + (id * H + ih) * W;
                                int w_row = w_base + (kd * Kh + kh) * Kw;

                                for (int kw = 0; kw < Kw; ++kw) {
                                    int iw = in_w0 + kw;
                                    if ((unsigned)iw >= (unsigned)W) continue;
                                    float xv = x[x_row + iw];
                                    float wv = w[w_row + kw];
                                    acc = fmaf(xv, wv, acc);
                                }
                            }
                        }
                    }

                    m = fmaxf(m, acc);
                }
            }
        }

        global_max = fmaxf(global_max, m);
    }

    // Second pass: accumulate exp(m - global_max)
    float sum_exp = 0.0f;
    for (int co = 0; co < Cout; ++co) {
        float m = -INFINITY;

        #pragma unroll
        for (int tz = 0; tz < 2; ++tz) {
            int od = od0 + tz;
            if ((unsigned)od >= (unsigned)Dc) continue;
            #pragma unroll
            for (int ty = 0; ty < 2; ++ty) {
                int oh = oh0 + ty;
                if ((unsigned)oh >= (unsigned)Hc) continue;
                #pragma unroll
                for (int tx = 0; tx < 2; ++tx) {
                    int ow_ = ow0 + tx;
                    if ((unsigned)ow_ >= (unsigned)Wc) continue;

                    int in_d0 = od * stride_d - pad_d;
                    int in_h0 = oh * stride_h - pad_h;
                    int in_w0 = ow_ * stride_w - pad_w;

                    float acc = (b != nullptr) ? b[co] : 0.0f;

                    for (int ci = 0; ci < Cin; ++ci) {
                        int w_base = ((co * Cin + ci) * Kd * Kh * Kw);
                        int x_base = ((n * Cin + ci) * D * H * W);

                        for (int kd = 0; kd < Kd; ++kd) {
                            int id = in_d0 + kd;
                            if ((unsigned)id >= (unsigned)D) continue;
                            for (int kh = 0; kh < Kh; ++kh) {
                                int ih = in_h0 + kh;
                                if ((unsigned)ih >= (unsigned)H) continue;

                                int x_row = x_base + (id * H + ih) * W;
                                int w_row = w_base + (kd * Kh + kh) * Kw;

                                for (int kw = 0; kw < Kw; ++kw) {
                                    int iw = in_w0 + kw;
                                    if ((unsigned)iw >= (unsigned)W) continue;
                                    float xv = x[x_row + iw];
                                    float wv = w[w_row + kw];
                                    acc = fmaf(xv, wv, acc);
                                }
                            }
                        }
                    }

                    m = fmaxf(m, acc);
                }
            }
        }

        sum_exp += expf(m - global_max);
    }

    float lse = global_max + logf(sum_exp);
    float outv = fmaxf(lse, 0.0f); // ReLU

    int y_idx = (((n * 1 + 0) * Dp + pd) * Hp + ph) * Wp + pw;
    y[y_idx] = outv;
}

torch::Tensor conv3d_maxpool_logsumexp_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,Cin,D,H,W)");
    TORCH_CHECK(w.dim() == 5, "w must be 5D (Cout,Cin,Kd,Kh,Kw)");
    TORCH_CHECK(x.size(1) == w.size(1), "Cin mismatch");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();

    const int N   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int D   = (int)x.size(2);
    const int H   = (int)x.size(3);
    const int W   = (int)x.size(4);

    const int Cout = (int)w.size(0);
    const int Kd   = (int)w.size(2);
    const int Kh   = (int)w.size(3);
    const int Kw   = (int)w.size(4);

    const int sd = (int)stride_d;
    const int sh = (int)stride_h;
    const int sw = (int)stride_w;
    const int pd = (int)pad_d;
    const int ph = (int)pad_h;
    const int pw = (int)pad_w;

    TORCH_CHECK(sd > 0 && sh > 0 && sw > 0, "stride must be > 0");
    TORCH_CHECK(pd >= 0 && ph >= 0 && pw >= 0, "padding must be >= 0");

    // conv output sizes (dilation=1)
    const int Dc = (D + 2*pd - Kd) / sd + 1;
    const int Hc = (H + 2*ph - Kh) / sh + 1;
    const int Wc = (W + 2*pw - Kw) / sw + 1;
    TORCH_CHECK(Dc > 0 && Hc > 0 && Wc > 0, "Invalid conv output size");

    // maxpool3d(kernel=2,stride=2,pad=0,dilation=1,ceil_mode=False)
    const int Dp = (Dc - 2) / 2 + 1;
    const int Hp = (Hc - 2) / 2 + 1;
    const int Wp = (Wc - 2) / 2 + 1;
    TORCH_CHECK(Dp > 0 && Hp > 0 && Wp > 0, "Invalid pooled output size");

    const float* b_ptr = nullptr;
    torch::Tensor b;
    if (b_opt.has_value() && b_opt.value().defined()) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA if provided");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "bias must be shape [Cout]");
        if (!b.is_contiguous()) b = b.contiguous();
        b_ptr = (const float*)b.data_ptr<float>();
    }

    auto y = torch::empty({N, 1, Dp, Hp, Wp}, x.options());

    const int total = N * Dp * Hp * Wp;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;

    conv3d_maxpool_logsumexp_relu_fwd_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)w.data_ptr<float>(),
        b_ptr,
        (float*)y.data_ptr<float>(),
        N, Cin, D, H, W,
        Cout, Kd, Kh, Kw,
        sd, sh, sw,
        pd, ph, pw,
        Dc, Hc, Wc,
        Dp, Hp, Wp
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv3d_maxpool_logsumexp_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_max_lse_relu",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3d_maxpool_logsumexp_relu_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Fused replacement for:
      conv3d -> maxpool3d(k=2,s=2) -> logsumexp(dim=1, keepdim=True) -> relu

    Assumptions:
      - CUDA float32
      - NCDHW contiguous
      - conv dilation=1, groups=1
      - maxpool: kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
      - forward only (no autograd kernel); suitable for inference / benchmarking
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        # Normalize stride/padding to 3D tuples
        def _to_3(v):
            if isinstance(v, int):
                return (v, v, v)
            if isinstance(v, (tuple, list)) and len(v) == 3:
                return (int(v[0]), int(v[1]), int(v[2]))
            raise ValueError("stride/padding must be int or tuple/list of len 3")

        if isinstance(kernel_size, int):
            k = (kernel_size, kernel_size, kernel_size)
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 3:
            k = (int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]))
        else:
            raise ValueError("kernel_size must be int or tuple/list of len 3")

        self.stride = _to_3(stride)
        self.padding = _to_3(padding)
        self.kernel_size = k
        self.custom_ops_lib = custom_ops_lib

        # Parameters in nn.Conv3d layout
        weight = torch.empty(out_channels, in_channels, k[0], k[1], k[2])
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        self.weight = nn.Parameter(weight)

        bias = torch.empty(out_channels)
        fan_in = in_channels * k[0] * k[1] * k[2]
        bound = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(bias, -bound, bound)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        b = self.bias

        if not w.is_cuda:
            w = w.cuda()
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        if not b.is_cuda:
            b = b.cuda()
        if b.dtype != torch.float32:
            b = b.float()
        if not b.is_contiguous():
            b = b.contiguous()

        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        return self.custom_ops_lib.conv3d_maxpool_logsumexp_relu_forward_cuda(
            x, w, b, sd, sh, sw, pd, ph, pw
        )