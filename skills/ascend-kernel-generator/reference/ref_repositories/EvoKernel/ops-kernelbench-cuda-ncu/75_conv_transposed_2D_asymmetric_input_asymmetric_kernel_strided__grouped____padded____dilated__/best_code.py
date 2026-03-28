import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ro_load(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ro_load(const float* p) { return *p; }
#endif

// ---------------- Generic kernel (fallback) ----------------
__global__ __launch_bounds__(256, 2)
void conv_transpose2d_forward_kernel_generic(
    const float* __restrict__ x,       // [N, Cin, Hin, Win]
    const float* __restrict__ w,       // [Cin, Cout_per_g, kH, kW]
    const float* __restrict__ b,       // [Cout] or nullptr
    float* __restrict__ y,             // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    int dH, int dW,
    int groups,
    int has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int tmp = idx / Wout;
    int oh = tmp % Hout;
    tmp /= Hout;
    int oc = tmp % Cout;
    int n  = tmp / Cout;

    const int Cout_per_g = Cout / groups;
    const int Cin_per_g  = Cin  / groups;

    int g = oc / Cout_per_g;
    int oc_in_g = oc - g * Cout_per_g;

    float acc = 0.0f;
    if (has_bias) acc = ro_load(b + oc);

    const int ic_start = g * Cin_per_g;
    const int ic_end   = ic_start + Cin_per_g;
    const int HW = Hin * Win;

    for (int kh = 0; kh < kH; ++kh) {
        int ih_num = oh + pH - kh * dH;
        if (ih_num % sH != 0) continue;
        int ih = ih_num / sH;
        if ((unsigned)ih >= (unsigned)Hin) continue;

        for (int kw = 0; kw < kW; ++kw) {
            int iw_num = ow + pW - kw * dW;
            if (iw_num % sW != 0) continue;
            int iw = iw_num / sW;
            if ((unsigned)iw >= (unsigned)Win) continue;

            const int x_hw = ih * Win + iw;
            const int x_base = (n * Cin) * HW + x_hw;
            const int w_k = kh * kW + kw;

            #pragma unroll 1
            for (int ic = ic_start; ic < ic_end; ++ic) {
                float xv = ro_load(x + x_base + ic * HW);
                int w_base = (ic * Cout_per_g + oc_in_g) * (kH * kW);
                float wv = ro_load(w + w_base + w_k);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[((n * Cout + oc) * Hout + oh) * Wout + ow] = acc;
}

// -------- Specialized kernel for benchmark configuration --------
// Fixed params: kH=3,kW=5,sH=2,sW=3,pH=1,pW=2,dH=2,dW=1,groups=4
// Each thread computes 1 (n,oh,ow) and 4 output channels in the same group.
// No shared memory; rely on RO cache for weights; more compute per thread to amortize x loads.
__global__ __launch_bounds__(128, 4)
void conv_transpose2d_forward_kernel_spec_3x5_s2x3_p1x2_d2x1_g4_oc4(
    const float* __restrict__ x,       // [N, Cin, Hin, Win]
    const float* __restrict__ w,       // [Cin, Cout_per_g, 3, 5]
    const float* __restrict__ b,       // [Cout] or nullptr
    float* __restrict__ y,             // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int has_bias,
    int z_base
) {
    constexpr int kH = 3, kW = 5;
    constexpr int sH = 2, sW = 3;
    constexpr int pH = 1, pW = 2;
    constexpr int dH = 2, dW = 1;
    constexpr int groups = 4;

    // block: (tx=ow lanes, ty=oc4 lanes)
    const int ow = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (ow >= Wout) return;

    const int oh = (int)blockIdx.y;
    if (oh >= Hout) return;

    // z spans (n, g, oc4_index_in_g)
    const int z = (int)blockIdx.z + z_base;

    const int Cout_per_g = Cout / groups;  // 16 in benchmark
    const int Cin_per_g  = Cin  / groups;  // 8 in benchmark
    const int oc4_per_g = (Cout_per_g + 3) >> 2; // 4 for benchmark
    const int z_per_n = groups * oc4_per_g;       // 16 for benchmark

    const int n = z / z_per_n;
    if (n >= N) return;
    const int zg = z - n * z_per_n;
    const int g = zg / oc4_per_g;
    const int oc4_idx_in_g = zg - g * oc4_per_g;

    const int oc0_in_g = oc4_idx_in_g * 4;
    const int oc0 = g * Cout_per_g + oc0_in_g;

    // Early-out for height alignment:
    // (oh + pH - kh*dH) % sH == 0; with sH=2,dH=2 => kh*dH even => need (oh+pH) even
    if (((oh + pH) & 1) != 0) {
        if (oc0 < Cout) {
            float v0 = has_bias ? ro_load(b + oc0) : 0.f;
            y[((n * Cout + oc0) * Hout + oh) * Wout + ow] = v0;
        }
        if (oc0 + 1 < Cout && oc0_in_g + 1 < Cout_per_g) {
            float v1 = has_bias ? ro_load(b + (oc0 + 1)) : 0.f;
            y[((n * Cout + (oc0 + 1)) * Hout + oh) * Wout + ow] = v1;
        }
        if (oc0 + 2 < Cout && oc0_in_g + 2 < Cout_per_g) {
            float v2 = has_bias ? ro_load(b + (oc0 + 2)) : 0.f;
            y[((n * Cout + (oc0 + 2)) * Hout + oh) * Wout + ow] = v2;
        }
        if (oc0 + 3 < Cout && oc0_in_g + 3 < Cout_per_g) {
            float v3 = has_bias ? ro_load(b + (oc0 + 3)) : 0.f;
            y[((n * Cout + (oc0 + 3)) * Hout + oh) * Wout + ow] = v3;
        }
        return;
    }

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const bool v1 = (oc0_in_g + 1) < Cout_per_g;
    const bool v2 = (oc0_in_g + 2) < Cout_per_g;
    const bool v3 = (oc0_in_g + 3) < Cout_per_g;

    if (has_bias) {
        acc0 = ro_load(b + oc0);
        if (v1) acc1 = ro_load(b + (oc0 + 1));
        if (v2) acc2 = ro_load(b + (oc0 + 2));
        if (v3) acc3 = ro_load(b + (oc0 + 3));
    }

    // Width mapping: need (ow + pW - kw) divisible by 3 => kw in {res, res+3}, res=(ow+pW)%3
    const int rW = ow + pW;
    const int res = rW - (rW / 3) * 3; // 0..2
    const int kwA = res;
    const int kwB = res + 3;
    const bool kwB_valid = (kwB < kW);

    const int iwA = (ow + pW - kwA) / sW;
    const int iwB = (ow + pW - kwB) / sW;

    const int HW = Hin * Win;
    const int ic_start = g * Cin_per_g;

    const int x_n_base = (n * Cin + ic_start) * HW;
    const float* __restrict__ x_base = x + x_n_base;

    // unroll kh
    #pragma unroll
    for (int kh = 0; kh < kH; ++kh) {
        const int ih = (oh + pH - kh * dH) / sH;
        if ((unsigned)ih >= (unsigned)Hin) continue;

        const int x_h_base = ih * Win;
        const int kkA = kh * kW + kwA;
        const int kkB = kh * kW + kwB;

        // A tap
        if ((unsigned)iwA < (unsigned)Win) {
            const int x_off = x_h_base + iwA;

            #pragma unroll
            for (int icg = 0; icg < 8; ++icg) { // Cin_per_g fixed to 8 in benchmark path
                const float xv = ro_load(x_base + icg * HW + x_off);
                const int ic = ic_start + icg;
                const int w_ic_base = ic * Cout_per_g * (kH * kW) + kkA;

                const float w0 = ro_load(w + (w_ic_base + (oc0_in_g + 0) * (kH * kW)));
                acc0 = fmaf(xv, w0, acc0);
                if (v1) { const float w1f = ro_load(w + (w_ic_base + (oc0_in_g + 1) * (kH * kW))); acc1 = fmaf(xv, w1f, acc1); }
                if (v2) { const float w2f = ro_load(w + (w_ic_base + (oc0_in_g + 2) * (kH * kW))); acc2 = fmaf(xv, w2f, acc2); }
                if (v3) { const float w3f = ro_load(w + (w_ic_base + (oc0_in_g + 3) * (kH * kW))); acc3 = fmaf(xv, w3f, acc3); }
            }
        }

        // B tap
        if (kwB_valid && (unsigned)iwB < (unsigned)Win) {
            const int x_off = x_h_base + iwB;

            #pragma unroll
            for (int icg = 0; icg < 8; ++icg) {
                const float xv = ro_load(x_base + icg * HW + x_off);
                const int ic = ic_start + icg;
                const int w_ic_base = ic * Cout_per_g * (kH * kW) + kkB;

                const float w0 = ro_load(w + (w_ic_base + (oc0_in_g + 0) * (kH * kW)));
                acc0 = fmaf(xv, w0, acc0);
                if (v1) { const float w1f = ro_load(w + (w_ic_base + (oc0_in_g + 1) * (kH * kW))); acc1 = fmaf(xv, w1f, acc1); }
                if (v2) { const float w2f = ro_load(w + (w_ic_base + (oc0_in_g + 2) * (kH * kW))); acc2 = fmaf(xv, w2f, acc2); }
                if (v3) { const float w3f = ro_load(w + (w_ic_base + (oc0_in_g + 3) * (kH * kW))); acc3 = fmaf(xv, w3f, acc3); }
            }
        }
    }

    // store
    y[((n * Cout + oc0) * Hout + oh) * Wout + ow] = acc0;
    if (v1) y[((n * Cout + (oc0 + 1)) * Hout + oh) * Wout + ow] = acc1;
    if (v2) y[((n * Cout + (oc0 + 2)) * Hout + oh) * Wout + ow] = acc2;
    if (v3) y[((n * Cout + (oc0 + 3)) * Hout + oh) * Wout + ow] = acc3;
}

torch::Tensor conv_transpose2d_asym_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW,
    int64_t groups
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be [Cin, Cout/groups, kH, kW]");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const bool has_bias = b_opt.has_value() && b_opt.value().defined();
    torch::Tensor b;
    const float* b_ptr = nullptr;
    if (has_bias) {
        b = b_opt.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.dim() == 1, "bias must be 1D [Cout]");
        b_ptr = b.data_ptr<float>();
    }

    int N   = (int)x_c.size(0);
    int Cin = (int)x_c.size(1);
    int Hin = (int)x_c.size(2);
    int Win = (int)x_c.size(3);

    TORCH_CHECK(Cin == (int)w_c.size(0), "weight.size(0) must equal Cin");
    TORCH_CHECK(Cin % (int)groups == 0, "Cin must be divisible by groups");

    int kH = (int)w_c.size(2);
    int kW = (int)w_c.size(3);

    int Cout_per_g = (int)w_c.size(1);
    int Cout = Cout_per_g * (int)groups;

    if (has_bias) TORCH_CHECK((int)b.size(0) == Cout, "bias size must match Cout");

    int Hout = (Hin - 1) * (int)sH - 2 * (int)pH + (int)dH * (kH - 1) + 1;
    int Wout = (Win - 1) * (int)sW - 2 * (int)pW + (int)dW * (kW - 1) + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "Invalid output size computed");

    auto y = torch::empty({N, Cout, Hout, Wout}, x_c.options());

    bool use_spec =
        (kH == 3 && kW == 5 &&
         (int)sH == 2 && (int)sW == 3 &&
         (int)pH == 1 && (int)pW == 2 &&
         (int)dH == 2 && (int)dW == 1 &&
         (int)groups == 4 &&
         (Cout_per_g == 16) && (Cin / (int)groups == 8)); // tie to benchmark for full unroll

    if (use_spec) {
        // Use 128 threads in x for coalesced W; z chunks over (n,g,oc4)
        const int threads = 128;
        dim3 block(threads, 1, 1);
        dim3 grid;
        grid.x = (unsigned)div_up_int(Wout, threads);
        grid.y = (unsigned)Hout;

        const int oc4_per_g = (Cout_per_g + 3) >> 2; // 4
        const int z_per_n = (int)groups * oc4_per_g; // 16
        unsigned long long z_total = (unsigned long long)N * (unsigned long long)z_per_n;

        const unsigned int Z_MAX = 65535u;
        unsigned long long z_base = 0;

        while (z_base < z_total) {
            unsigned long long remaining = z_total - z_base;
            unsigned int z_chunk = (remaining > (unsigned long long)Z_MAX) ? Z_MAX : (unsigned int)remaining;
            grid.z = z_chunk;

            conv_transpose2d_forward_kernel_spec_3x5_s2x3_p1x2_d2x1_g4_oc4<<<grid, block, 0>>>(
                x_c.data_ptr<float>(),
                w_c.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, Cin, Hin, Win,
                Cout, Hout, Wout,
                has_bias ? 1 : 0,
                (int)z_base
            );
            z_base += z_chunk;
        }
    } else {
        int total = N * Cout * Hout * Wout;
        int threads = 256;
        int blocks = div_up_int(total, threads);
        conv_transpose2d_forward_kernel_generic<<<blocks, threads>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            y.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, Hout, Wout,
            kH, kW,
            (int)sH, (int)sW,
            (int)pH, (int)pW,
            (int)dH, (int)dW,
            (int)groups,
            has_bias ? 1 : 0
        );
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose2d_asym_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_asym_opt6",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose2d_asym_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    ConvTranspose2d replacement using a custom CUDA kernel (forward-only).
    Weight layout: [Cin, Cout/groups, kH, kW] (PyTorch ConvTranspose2d layout).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        kH, kW = kernel_size
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kH, self.kW = int(kH), int(kW)
        self.sH, self.sW = int(stride[0]), int(stride[1])
        self.pH, self.pW = int(padding[0]), int(padding[1])
        self.dH, self.dW = int(dilation[0]), int(dilation[1])
        self.groups = int(groups)

        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        w = torch.empty(
            self.in_channels,
            self.out_channels // self.groups,
            self.kH,
            self.kW,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
        else:
            self.bias = None

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
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

        return self.custom_ops.conv_transpose2d_asym_cuda(
            x, w, b,
            self.sH, self.sW,
            self.pH, self.pW,
            self.dH, self.dW,
            self.groups,
        )