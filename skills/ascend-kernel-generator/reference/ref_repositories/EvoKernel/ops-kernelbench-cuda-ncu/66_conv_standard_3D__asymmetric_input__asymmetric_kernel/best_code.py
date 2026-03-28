import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------ CUDA/C++ Extension: 3D Conv (asymmetric kernel) ------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

static inline int calc_out_int(int in, int k, int pad, int stride, int dil) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// ---------------- Generic kernel (grid-stride) ----------------
__global__ void conv3d_forward_f32_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,  // can be nullptr
    float* __restrict__ y,
    int N, int Cin, int Din, int Hin, int Win,
    int Cout,
    int Kd, int Kh, int Kw,
    int Sd, int Sh, int Sw,
    int Pd, int Ph, int Pw,
    int Ddil, int Hdil, int Wdil,
    int groups,
    int Dout, int Hout, int Wout
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * Cout * Dout * Hout * Wout;
    long long gstride = (long long)gridDim.x * blockDim.x;

    int cin_per_group  = Cin / groups;
    int cout_per_group = Cout / groups;

    for (long long linear = tid; linear < total; linear += gstride) {
        long long t = linear;
        int ow = (int)(t % Wout); t /= Wout;
        int oh = (int)(t % Hout); t /= Hout;
        int od = (int)(t % Dout); t /= Dout;
        int oc = (int)(t % Cout); t /= Cout;
        int n  = (int)t;

        int g = oc / cout_per_group;
        int ic_begin = g * cin_per_group;

        float acc = (b != nullptr) ? ldg_f32(b + oc) : 0.0f;

        int id0 = od * Sd - Pd;
        int ih0 = oh * Sh - Ph;
        int iw0 = ow * Sw - Pw;

        long long x_n_base = (long long)n * Cin * Din * Hin * Win;
        long long w_oc_base = (long long)oc * cin_per_group * Kd * Kh * Kw;

        for (int icg = 0; icg < cin_per_group; ++icg) {
            int ic = ic_begin + icg;
            long long x_c_base = x_n_base + (long long)ic * Din * Hin * Win;
            long long w_ic_base = w_oc_base + (long long)icg * Kd * Kh * Kw;

#pragma unroll 1
            for (int kd = 0; kd < Kd; ++kd) {
                int id = id0 + kd * Ddil;
                if ((unsigned)id >= (unsigned)Din) continue;
                long long x_d_base = x_c_base + (long long)id * Hin * Win;
                long long w_kd_base = w_ic_base + (long long)kd * Kh * Kw;

#pragma unroll 1
                for (int kh = 0; kh < Kh; ++kh) {
                    int ih = ih0 + kh * Hdil;
                    if ((unsigned)ih >= (unsigned)Hin) continue;
                    long long x_h_base = x_d_base + (long long)ih * Win;
                    long long w_kh_base = w_kd_base + (long long)kh * Kw;

#pragma unroll 1
                    for (int kw = 0; kw < Kw; ++kw) {
                        int iw = iw0 + kw * Wdil;
                        if ((unsigned)iw >= (unsigned)Win) continue;
                        float xv = ldg_f32(x + (x_h_base + iw));
                        float wv = ldg_f32(w + (w_kh_base + kw));
                        acc = fmaf(xv, wv, acc);
                    }
                }
            }
        }

        y[((((long long)n * Cout + oc) * Dout + od) * Hout + oh) * Wout + ow] = acc;
    }
}

// ---------------- Weight packing kernel: W [Cout,3,3,5,7] -> Wpack [Cout,3,5,7,float4] ----------------
// Layout: wpack[oc, kd, kh, kw] = float4(w[oc,0,kd,kh,kw], w[oc,1,...], w[oc,2,...], 0)
__global__ void pack_w_k357_cin3_to_float4_kernel(
    const float* __restrict__ w,
    float4* __restrict__ wpack,
    int Cout
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)Cout * 3LL * 5LL * 7LL;
    long long stride = (long long)gridDim.x * blockDim.x;

    constexpr int Kd=3, Kh=5, Kw=7;
    constexpr int kHW = Kh*Kw;      // 35
    constexpr int kDHW = Kd*kHW;    // 105

    for (long long idx = tid; idx < total; idx += stride) {
        long long t = idx;
        int kw = (int)(t % Kw); t /= Kw;
        int kh = (int)(t % Kh); t /= Kh;
        int kd = (int)(t % Kd); t /= Kd;
        int oc = (int)t;

        // source offsets in original weights
        long long base_oc = (long long)oc * 3LL * kDHW;
        long long off = (long long)kd * kHW + (long long)kh * Kw + kw;

        float a0 = ldg_f32(w + base_oc + 0LL * kDHW + off);
        float a1 = ldg_f32(w + base_oc + 1LL * kDHW + off);
        float a2 = ldg_f32(w + base_oc + 2LL * kDHW + off);
        float4 v = make_float4(a0, a1, a2, 0.0f);

        long long dst = (((long long)oc * Kd + kd) * Kh + kh) * Kw + kw; // contiguous float4 taps
        wpack[dst] = v;
    }
}

// ---------------- Fast path: groups=1, stride=1, dilation=1, K=(3,5,7), Cin=3, padding=0 ----------------
// Each thread computes 4 output channels for one spatial point.
// Uses packed weights: wpack [Cout, 3,5,7] as float4 per tap for Cin=3.
__global__ __launch_bounds__(128, 3)
void conv3d_forward_f32_k357_s111_d111_g1_cin3_p0_oc4_wpack_kernel(
    const float* __restrict__ x,
    const float4* __restrict__ wpack,
    const float* __restrict__ b,  // can be nullptr
    float* __restrict__ y,
    int N, int Din, int Hin, int Win,
    int Cout,
    int Dout, int Hout, int Wout
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_spatial = (long long)N * Dout * Hout * Wout;
    long long gstride = (long long)gridDim.x * blockDim.x;

    int oc0 = (int)blockIdx.y * 4;
    if (oc0 >= Cout) return;
    int oc1 = oc0 + 1, oc2 = oc0 + 2, oc3 = oc0 + 3;
    bool h1 = oc1 < Cout, h2 = oc2 < Cout, h3 = oc3 < Cout;

    constexpr int Kd=3, Kh=5, Kw=7;
    constexpr int kHW = Kh*Kw;      // 35

    const long long HW = (long long)Hin * Win;
    const long long DHW = (long long)Din * HW;
    const long long plane = (long long)Dout * Hout * Wout;

    for (long long linear = tid; linear < total_spatial; linear += gstride) {
        long long t = linear;
        int ow = (int)(t % Wout); t /= Wout;
        int oh = (int)(t % Hout); t /= Hout;
        int od = (int)(t % Dout); t /= Dout;
        int n  = (int)t;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        if (b != nullptr) {
            acc0 = ldg_f32(b + oc0);
            if (h1) acc1 = ldg_f32(b + oc1);
            if (h2) acc2 = ldg_f32(b + oc2);
            if (h3) acc3 = ldg_f32(b + oc3);
        }

        // Base pointers
        const float* x0 = x + (long long)n * 3LL * DHW; // ic0 base
        const float* x1 = x0 + DHW;
        const float* x2 = x1 + DHW;

        // output indexing (contiguous NCDHW): y[((n*Cout+oc)*Dout+od)*Hout+oh)*Wout+ow]
        long long out_sp = (((long long)od * Hout + oh) * Wout + ow);
        float* y_n = y + (long long)n * (long long)Cout * plane;

        // weight base indices for each oc
        const float4* w0 = wpack + (long long)oc0 * (Kd*Kh*Kw);
        const float4* w1 = wpack + (long long)oc1 * (Kd*Kh*Kw);
        const float4* w2 = wpack + (long long)oc2 * (Kd*Kh*Kw);
        const float4* w3 = wpack + (long long)oc3 * (Kd*Kh*Kw);

#pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            int id = od + kd;
            const float* x0d = x0 + (long long)id * HW;
            const float* x1d = x1 + (long long)id * HW;
            const float* x2d = x2 + (long long)id * HW;

#pragma unroll
            for (int kh = 0; kh < 5; ++kh) {
                int ih = oh + kh;
                long long x_row = (long long)ih * Win + ow;

                const float* px0 = x0d + x_row;
                const float* px1 = x1d + x_row;
                const float* px2 = x2d + x_row;

                int tap_base = (kd * 5 + kh) * 7;

#pragma unroll
                for (int kw = 0; kw < 7; ++kw) {
                    float xv0 = ldg_f32(px0 + kw);
                    float xv1 = ldg_f32(px1 + kw);
                    float xv2 = ldg_f32(px2 + kw);

                    int tap = tap_base + kw;

                    float4 v0 = w0[tap];
                    acc0 = fmaf(xv0, v0.x, acc0);
                    acc0 = fmaf(xv1, v0.y, acc0);
                    acc0 = fmaf(xv2, v0.z, acc0);

                    if (h1) {
                        float4 v1 = w1[tap];
                        acc1 = fmaf(xv0, v1.x, acc1);
                        acc1 = fmaf(xv1, v1.y, acc1);
                        acc1 = fmaf(xv2, v1.z, acc1);
                    }
                    if (h2) {
                        float4 v2 = w2[tap];
                        acc2 = fmaf(xv0, v2.x, acc2);
                        acc2 = fmaf(xv1, v2.y, acc2);
                        acc2 = fmaf(xv2, v2.z, acc2);
                    }
                    if (h3) {
                        float4 v3 = w3[tap];
                        acc3 = fmaf(xv0, v3.x, acc3);
                        acc3 = fmaf(xv1, v3.y, acc3);
                        acc3 = fmaf(xv2, v3.z, acc3);
                    }
                }
            }
        }

        // stores
        y_n[(((long long)oc0) * plane) + out_sp] = acc0;
        if (h1) y_n[(((long long)oc1) * plane) + out_sp] = acc1;
        if (h2) y_n[(((long long)oc2) * plane) + out_sp] = acc2;
        if (h3) y_n[(((long long)oc3) * plane) + out_sp] = acc3;
    }
}

// Tail kernel: for remaining oc (Cout not multiple of 4), still uses wpack but computes 1 oc per thread
__global__ __launch_bounds__(256, 2)
void conv3d_forward_f32_k357_s111_d111_g1_cin3_p0_oc1_wpack_tail_kernel(
    const float* __restrict__ x,
    const float4* __restrict__ wpack,
    const float* __restrict__ b,  // can be nullptr
    float* __restrict__ y,
    int N, int Din, int Hin, int Win,
    int Cout,
    int oc_start,
    int Dout, int Hout, int Wout
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * (long long)(Cout - oc_start) * Dout * Hout * Wout;
    long long gstride = (long long)gridDim.x * blockDim.x;

    constexpr int Kd=3, Kh=5, Kw=7;
    const long long HW = (long long)Hin * Win;
    const long long DHW = (long long)Din * HW;
    const long long plane = (long long)Dout * Hout * Wout;

    for (long long linear = tid; linear < total; linear += gstride) {
        long long t = linear;
        int ow = (int)(t % Wout); t /= Wout;
        int oh = (int)(t % Hout); t /= Hout;
        int od = (int)(t % Dout); t /= Dout;
        int oc_rel = (int)(t % (Cout - oc_start)); t /= (Cout - oc_start);
        int n = (int)t;
        int oc = oc_start + oc_rel;

        float acc = (b != nullptr) ? ldg_f32(b + oc) : 0.f;

        const float* x0 = x + (long long)n * 3LL * DHW;
        const float* x1 = x0 + DHW;
        const float* x2 = x1 + DHW;

        const float4* w0 = wpack + (long long)oc * (Kd*Kh*Kw);

#pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            int id = od + kd;
            const float* x0d = x0 + (long long)id * HW;
            const float* x1d = x1 + (long long)id * HW;
            const float* x2d = x2 + (long long)id * HW;

#pragma unroll
            for (int kh = 0; kh < 5; ++kh) {
                int ih = oh + kh;
                long long x_row = (long long)ih * Win + ow;

                const float* px0 = x0d + x_row;
                const float* px1 = x1d + x_row;
                const float* px2 = x2d + x_row;

                int tap_base = (kd * 5 + kh) * 7;

#pragma unroll
                for (int kw = 0; kw < 7; ++kw) {
                    float xv0 = ldg_f32(px0 + kw);
                    float xv1 = ldg_f32(px1 + kw);
                    float xv2 = ldg_f32(px2 + kw);
                    float4 v = w0[tap_base + kw];
                    acc = fmaf(xv0, v.x, acc);
                    acc = fmaf(xv1, v.y, acc);
                    acc = fmaf(xv2, v.z, acc);
                }
            }
        }

        float* y_n = y + (long long)n * (long long)Cout * plane;
        long long out_sp = (((long long)od * Hout + oh) * Wout + ow);
        y_n[(long long)oc * plane + out_sp] = acc;
    }
}

torch::Tensor conv_standard3d_asymmetric_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    std::vector<int64_t> stride_dhw,
    std::vector<int64_t> pad_dhw,
    std::vector<int64_t> dil_dhw,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be NCDHW");
    TORCH_CHECK(w.dim() == 5, "w must be [Cout, Cin/groups, Kd, Kh, Kw]");
    TORCH_CHECK(stride_dhw.size() == 3, "stride must be 3D");
    TORCH_CHECK(pad_dhw.size() == 3, "pad must be 3D");
    TORCH_CHECK(dil_dhw.size() == 3, "dilation must be 3D");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();

    torch::Tensor b;
    const float* b_ptr = nullptr;
    if (b_opt.has_value() && b_opt.value().defined() && b_opt.value().numel() > 0) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1, "bias must be 1D [Cout]");
        if (!b.is_contiguous()) b = b.contiguous();
        b_ptr = (const float*)b.data_ptr<float>();
    }

    int N   = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Din = (int)x.size(2);
    int Hin = (int)x.size(3);
    int Win = (int)x.size(4);

    int Cout = (int)w.size(0);
    int CinG = (int)w.size(1);
    int Kd   = (int)w.size(2);
    int Kh   = (int)w.size(3);
    int Kw   = (int)w.size(4);

    TORCH_CHECK(Cin % (int)groups == 0, "Cin must be divisible by groups");
    TORCH_CHECK(Cout % (int)groups == 0, "Cout must be divisible by groups");
    TORCH_CHECK(CinG == Cin / (int)groups, "w.size(1) must equal Cin/groups");

    int Sd = (int)stride_dhw[0], Sh = (int)stride_dhw[1], Sw = (int)stride_dhw[2];
    int Pd = (int)pad_dhw[0],    Ph = (int)pad_dhw[1],    Pw = (int)pad_dhw[2];
    int Ddil = (int)dil_dhw[0],  Hdil = (int)dil_dhw[1],  Wdil = (int)dil_dhw[2];

    int Dout = calc_out_int(Din, Kd, Pd, Sd, Ddil);
    int Hout = calc_out_int(Hin, Kh, Ph, Sh, Hdil);
    int Wout = calc_out_int(Win, Kw, Pw, Sw, Wdil);
    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "Invalid output shape computed");

    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x.options());

    const float* x_ptr = (const float*)x.data_ptr<float>();
    const float* w_ptr = (const float*)w.data_ptr<float>();
    float* y_ptr = (float*)y.data_ptr<float>();

    int g = (int)groups;

    // Fast path: groups=1, Cin=3, stride=1, dilation=1, K=(3,5,7), padding=0
    bool fast_p0 =
        (g == 1) &&
        (Cin == 3) &&
        (CinG == 3) &&
        (Sd == 1 && Sh == 1 && Sw == 1) &&
        (Ddil == 1 && Hdil == 1 && Wdil == 1) &&
        (Kd == 3 && Kh == 5 && Kw == 7) &&
        (Pd == 0 && Ph == 0 && Pw == 0);

    if (fast_p0) {
        // pack weights to float4 taps
        auto wpack = torch::empty({Cout, 3, 5, 7, 4}, w.options()); // float32 storage; viewed as float4
        float4* wpack_ptr = (float4*)wpack.data_ptr<float>();

        {
            long long total = (long long)Cout * 3LL * 5LL * 7LL;
            int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            pack_w_k357_cin3_to_float4_kernel<<<blocks, threads>>>(w_ptr, wpack_ptr, Cout);
        }

        // main OCx4 kernel
        int oc_quads = Cout / 4;
        if (oc_quads > 0) {
            long long total_sp = (long long)N * Dout * Hout * Wout;
            int threads = 128;
            int blocks_x = (int)((total_sp + threads - 1) / threads);
            if (blocks_x > 65535) blocks_x = 65535;
            dim3 grid((unsigned)blocks_x, (unsigned)oc_quads, 1);

            conv3d_forward_f32_k357_s111_d111_g1_cin3_p0_oc4_wpack_kernel<<<grid, threads>>>(
                x_ptr, (const float4*)wpack_ptr, b_ptr, y_ptr,
                N, Din, Hin, Win,
                Cout,
                Dout, Hout, Wout
            );
        }

        // tail channels
        int oc_start = (Cout / 4) * 4;
        if (oc_start < Cout) {
            long long total_tail = (long long)N * (long long)(Cout - oc_start) * Dout * Hout * Wout;
            int threads = 256;
            int blocks = (int)((total_tail + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            conv3d_forward_f32_k357_s111_d111_g1_cin3_p0_oc1_wpack_tail_kernel<<<blocks, threads>>>(
                x_ptr, (const float4*)wpack_ptr, b_ptr, y_ptr,
                N, Din, Hin, Win,
                Cout, oc_start,
                Dout, Hout, Wout
            );
        }

        return y;
    }

    // Generic fallback
    long long total = (long long)N * Cout * Dout * Hout * Wout;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    conv3d_forward_f32_generic_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, y_ptr,
        N, Cin, Din, Hin, Win,
        Cout,
        Kd, Kh, Kw,
        Sd, Sh, Sw,
        Pd, Ph, Pw,
        Ddil, Hdil, Wdil,
        g,
        Dout, Hout, Wout
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv_standard3d_asymmetric_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    std::vector<int64_t> stride_dhw,
    std::vector<int64_t> pad_dhw,
    std::vector<int64_t> dil_dhw,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_asym_opt4_wpack_oc4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv_standard3d_asymmetric_input_asymmetric_kernel_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ------------------ Model using the custom op ------------------

class ModelNew(nn.Module):
    """
    3D convolution using a custom CUDA kernel (forward only), supporting asymmetric kernels.
    Fast path specialized for groups=1, Cin=3, k=(3,5,7), stride=1, dilation=1, padding=0.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

        assert len(kernel_size) == 3
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]))
        self.stride = (int(stride[0]), int(stride[1]), int(stride[2]))
        self.padding = (int(padding[0]), int(padding[1]), int(padding[2]))
        self.dilation = (int(dilation[0]), int(dilation[1]), int(dilation[2]))
        self.groups = int(groups)

        cin_g = self.in_channels // self.groups
        kd, kh, kw = self.kernel_size

        w = torch.empty(self.out_channels, cin_g, kd, kh, kw, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.zeros(self.out_channels, dtype=torch.float32)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()

        w = self.weight
        b = self.bias
        if not w.is_cuda:
            w = w.cuda()
        if b is not None and (not b.is_cuda):
            b = b.cuda()

        return self.custom_ops_lib.conv_standard3d_asymmetric_input_asymmetric_kernel_cuda(
            x, w, b,
            [self.stride[0], self.stride[1], self.stride[2]],
            [self.padding[0], self.padding[1], self.padding[2]],
            [self.dilation[0], self.dilation[1], self.dilation[2]],
            int(self.groups),
        )