import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __forceinline__ __device__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline __host__ __device__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

// -------------------------
// Generic fallback kernel
// -------------------------
__global__ void conv_transpose2d_forward_generic_kernel(
    const float* __restrict__ x,       // [N, Cin, Hin, Win]
    const float* __restrict__ w,       // [Cin, Cout, kH, kW]
    const float* __restrict__ b,       // [Cout] or nullptr
    float* __restrict__ y,             // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int kH, int kW,
    int stride, int padding, int dilation,
    int Hout, int Wout,
    bool has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long total = (long)N * (long)Cout * (long)Hout * (long)Wout;
    if ((long)idx >= total) return;

    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int oc = (idx / (Wout * Hout)) % Cout;
    int n  = idx / (Wout * Hout * Cout);

    float acc = has_bias ? ldg_f(b + oc) : 0.0f;

    for (int ic = 0; ic < Cin; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            int num_h = oh + padding - kh * dilation;
            if (num_h % stride != 0) continue;
            int ih = num_h / stride;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int num_w = ow + padding - kw * dilation;
                if (num_w % stride != 0) continue;
                int iw = num_w / stride;
                if ((unsigned)iw >= (unsigned)Win) continue;

                int x_off = ((n * Cin + ic) * Hin + ih) * Win + iw;
                int w_off = ((ic * Cout + oc) * kH + kh) * kW + kw;
                acc = fmaf(ldg_f(x + x_off), ldg_f(w + w_off), acc);
            }
        }
    }

    int y_off = ((n * Cout + oc) * Hout + oh) * Wout + ow;
    y[y_off] = acc;
}

// -------------------------
// Specialized kernel for Cin=32,Cout=64,k=3,stride=5,dilation=2,padding=1
// Each thread computes 2 ow positions (ow, ow+1) for same (n,oc,oh)
// -------------------------
__device__ __forceinline__ void map_coord_s5_d2_p1(int coord, int in_lim, int &i0, int &k0, bool &valid) {
    // Solve: (coord + 1 - 2*k) % 5 == 0 for k in {0,1,2}
    // r=(coord+1)%5
    // r=0 -> k=0, i=(coord+1)/5
    // r=2 -> k=1, i=(coord-1)/5
    // r=4 -> k=2, i=(coord-3)/5
    // else invalid
    int r = (coord + 1) % 5;
    if (r == 0) { k0 = 0; i0 = (coord + 1) / 5; valid = ((unsigned)i0 < (unsigned)in_lim); }
    else if (r == 2) { k0 = 1; i0 = (coord - 1) / 5; valid = ((unsigned)i0 < (unsigned)in_lim); }
    else if (r == 4) { k0 = 2; i0 = (coord - 3) / 5; valid = ((unsigned)i0 < (unsigned)in_lim); }
    else { k0 = -1; i0 = -1; valid = false; }
}

__global__ __launch_bounds__(256, 2)
void conv_transpose2d_forward_s32_64_k3_s5_d2_p1_vec2_kernel(
    const float* __restrict__ x,   // [N, 32, Hin, Win]
    const float* __restrict__ w,   // [32, 64, 3, 3]
    const float* __restrict__ b,   // [64] or nullptr
    float* __restrict__ y,         // [N, 64, Hout, Wout]
    int N, int Hin, int Win,
    int Hout, int Wout,
    bool has_bias
) {
    // blockIdx.x tiles W in units of (blockDim.x*2)
    int tid = (int)threadIdx.x;
    int ow_base = ((int)blockIdx.x * (int)blockDim.x + tid) * 2;
    if ((unsigned)ow_base >= (unsigned)Wout) return;

    int n_oc = (int)blockIdx.y; // 0 .. N*64-1
    int oc = n_oc & 63;
    int n  = n_oc >> 6;
    if (n >= N) return;

    int oh = (int)blockIdx.z;
    if (oh >= Hout) return;

    float acc0 = has_bias ? ldg_f(b + oc) : 0.0f;
    float acc1 = acc0;

    // Map height once
    int ih, kh;
    bool vh;
    map_coord_s5_d2_p1(oh, Hin, ih, kh, vh);
    if (!vh) {
        // only bias
        int y_off0 = ((n * 64 + oc) * Hout + oh) * Wout + ow_base;
        y[y_off0] = acc0;
        if (ow_base + 1 < Wout) y[y_off0 + 1] = acc1;
        return;
    }

    // Map width for ow0 and ow1
    int iw0, kw0; bool vw0;
    map_coord_s5_d2_p1(ow_base, Win, iw0, kw0, vw0);

    int iw1, kw1; bool vw1;
    bool has_ow1 = (ow_base + 1) < Wout;
    if (has_ow1) map_coord_s5_d2_p1(ow_base + 1, Win, iw1, kw1, vw1);
    else { iw1=-1; kw1=-1; vw1=false; }

    if (!(vw0 | vw1)) {
        int y_off0 = ((n * 64 + oc) * Hout + oh) * Wout + ow_base;
        y[y_off0] = acc0;
        if (has_ow1) y[y_off0 + 1] = acc1;
        return;
    }

    // Pointers
    int x_n_base = n * 32 * Hin * Win;
    int x_h_base = x_n_base + ih * Win;  // add ic*Hin*Win inside loop
    int w_oc_base = oc * 9;

#pragma unroll
    for (int ic = 0; ic < 32; ++ic) {
        const float* __restrict__ w_ic = w + (ic * 64 * 9 + w_oc_base);
        int x_ic_base = x_h_base + ic * Hin * Win;

        // For fixed kh, only one row of weights contributes for this oh
        const float* __restrict__ w_row = w_ic + kh * 3;

        if (vw0) {
            float xv = ldg_f(x + (x_ic_base + iw0));
            float wv = ldg_f(w_row + kw0);
            acc0 = fmaf(xv, wv, acc0);
        }
        if (vw1) {
            float xv = ldg_f(x + (x_ic_base + iw1));
            float wv = ldg_f(w_row + kw1);
            acc1 = fmaf(xv, wv, acc1);
        }
    }

    int y_off0 = ((n * 64 + oc) * Hout + oh) * Wout + ow_base;
    y[y_off0] = acc0;
    if (has_ow1) y[y_off0 + 1] = acc1;
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [Cin, Cout, kH, kW]");

    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(dilation > 0, "dilation must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    const int N   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int Hin = (int)x.size(2);
    const int Win = (int)x.size(3);

    TORCH_CHECK((int)weight.size(0) == Cin, "weight.size(0) (Cin) must match x.size(1)");
    const int Cout = (int)weight.size(1);
    const int kH   = (int)weight.size(2);
    const int kW   = (int)weight.size(3);

    const int Hout = (int)((Hin - 1) * (int)stride - 2 * (int)padding + (int)dilation * (kH - 1) + 1);
    const int Wout = (int)((Win - 1) * (int)stride - 2 * (int)padding + (int)dilation * (kW - 1) + 1);
    TORCH_CHECK(Hout > 0 && Wout > 0, "computed output size is non-positive");

    const bool has_bias = bias.has_value() && bias.value().defined();
    const float* bptr = nullptr;
    if (has_bias) {
        auto b = bias.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.dim() == 1, "bias must be 1D [Cout]");
        TORCH_CHECK((int)b.size(0) == Cout, "bias.size(0) must match Cout");
        bptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    const bool use_specialized =
        (Cin == 32) && (Cout == 64) && (kH == 3) && (kW == 3) &&
        ((int)stride == 5) && ((int)dilation == 2) && ((int)padding == 1);

    if (use_specialized) {
        const int threads = 256;
        dim3 block(threads, 1, 1);
        const int grid_x = div_up_int(Wout, threads * 2);
        const int grid_y = N * 64;
        const int grid_z = Hout; // one oh per z for simpler control flow
        dim3 grid((unsigned)grid_x, (unsigned)grid_y, (unsigned)grid_z);

        conv_transpose2d_forward_s32_64_k3_s5_d2_p1_vec2_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bptr,
            y.data_ptr<float>(),
            N, Hin, Win,
            Hout, Wout,
            has_bias
        );
        return y;
    }

    // Generic fallback
    const int threads = 256;
    const int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)Hout * (int64_t)Wout;
    const int blocks = (int)((total + threads - 1) / threads);

    conv_transpose2d_forward_generic_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, kH, kW,
        (int)stride, (int)padding, (int)dilation,
        Hout, Wout,
        has_bias
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_opt3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose2d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement for nn.ConvTranspose2d using a custom CUDA kernel (forward-only).
    Assumes input is CUDA, contiguous, and float32 (casts/contiguous enforced).
    Weight layout expected: [Cin, Cout, kH, kW] (PyTorch ConvTranspose2d parameter layout).
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
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        k = int(kernel_size)
        self.kH = k
        self.kW = k

        w = torch.empty(in_channels, out_channels, self.kH, self.kW, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None

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

        b = self.bias
        if b is not None:
            if not b.is_cuda:
                b = b.to(device=x.device)
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()

        return self.custom_ops.conv_transpose2d_forward_cuda(
            x, w, b, self.stride, self.padding, self.dilation
        )