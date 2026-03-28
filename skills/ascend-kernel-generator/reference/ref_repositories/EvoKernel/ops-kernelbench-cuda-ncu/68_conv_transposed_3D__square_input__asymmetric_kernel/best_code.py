import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

// ------------------------ Constant-memory weights (optional fast path) ------------------------
// Max constant memory is 64KB. We store up to 16384 floats (64KB).
// For benchmark: Cin=32,Cout=64,K=75 => 153600 floats -> does NOT fit.
// But many real cases (or smaller Cout) could fit; keep logic guarded.
// Additionally, we support a "Cout tile" not used here; constant path used only when fits fully.
#ifndef CONST_W_MAX
#define CONST_W_MAX 16384
#endif
__constant__ float const_w[CONST_W_MAX];
__device__ __forceinline__ float LDW_CONST(int idx) { return const_w[idx]; }

static __device__ __forceinline__ int64_t idx_y_ncdhw(
    int n, int c, int d, int h, int w,
    int C, int D, int H, int W
) {
    return (((((int64_t)n * C + c) * D + d) * H + h) * W + w);
}

// w layout: [Cin, Cout, Kd, Kh, Kw]
static __device__ __forceinline__ int64_t idx_w_cicokdkhkw(
    int ci, int co, int kd, int kh, int kw,
    int Cout, int Kd, int Kh, int Kw
) {
    return (((((int64_t)ci * Cout + co) * Kd + kd) * Kh + kh) * Kw + kw);
}

// ------------------------ Baseline generic (kept) ------------------------
__global__ __launch_bounds__(128, 3)
void conv_t3d_s1p0_generic_co2(
    const float* __restrict__ x,   // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w,   // [Cin,Cout,Kd,Kh,Kw]
    float* __restrict__ y,         // [N,Cout,Dout,Hout,Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Kd, int Kh, int Kw,
    int Dout, int Hout, int Wout
) {
    const int64_t spatial = (int64_t)Dout * Hout * Wout;
    const int co_pairs = (Cout + 1) >> 1;
    const int64_t total = (int64_t)N * co_pairs * spatial;

    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = linear;

        int ow = (int)(t % Wout); t /= Wout;
        int oh = (int)(t % Hout); t /= Hout;
        int od = (int)(t % Dout); t /= Dout;

        int co_pair = (int)(t % co_pairs); t /= co_pairs;
        int n = (int)t;

        int co0 = co_pair * 2;
        int co1 = co0 + 1;
        bool v1 = (co1 < Cout);

        float acc0 = 0.0f, acc1 = 0.0f;

        int kd_min = od - (Din - 1); if (kd_min < 0) kd_min = 0;
        int kd_max = od; if (kd_max > (Kd - 1)) kd_max = (Kd - 1);

        int kh_min = oh - (Hin - 1); if (kh_min < 0) kh_min = 0;
        int kh_max = oh; if (kh_max > (Kh - 1)) kh_max = (Kh - 1);

        int kw_min = ow - (Win - 1); if (kw_min < 0) kw_min = 0;
        int kw_max = ow; if (kw_max > (Kw - 1)) kw_max = (Kw - 1);

        for (int ci = 0; ci < Cin; ++ci) {
            const float* __restrict__ x_base = x + ((int64_t)n * Cin + ci) * (int64_t)Din * Hin * Win;

            const float* __restrict__ w0_base = w + ((int64_t)ci * Cout + co0) * (int64_t)Kd * Kh * Kw;
            const float* __restrict__ w1_base = v1 ? (w + ((int64_t)ci * Cout + co1) * (int64_t)Kd * Kh * Kw) : nullptr;

            for (int kd = kd_min; kd <= kd_max; ++kd) {
                int id = od - kd;
                const float* __restrict__ x_d = x_base + (int64_t)id * Hin * Win;
                const float* __restrict__ w0_kd = w0_base + (int64_t)kd * Kh * Kw;
                const float* __restrict__ w1_kd = v1 ? (w1_base + (int64_t)kd * Kh * Kw) : nullptr;

                for (int kh = kh_min; kh <= kh_max; ++kh) {
                    int ih = oh - kh;
                    const float* __restrict__ x_dh = x_d + (int64_t)ih * Win;
                    const float* __restrict__ w0_kh = w0_kd + (int64_t)kh * Kw;
                    const float* __restrict__ w1_kh = v1 ? (w1_kd + (int64_t)kh * Kw) : nullptr;

                    #pragma unroll 1
                    for (int kw = kw_min; kw <= kw_max; ++kw) {
                        int iw = ow - kw;
                        float xv = LDG(x_dh + iw);
                        acc0 = fmaf(xv, LDG(w0_kh + kw), acc0);
                        if (v1) acc1 = fmaf(xv, LDG(w1_kh + kw), acc1);
                    }
                }
            }
        }

        const int64_t out_base = idx_y_ncdhw(n, co0, od, oh, ow, Cout, Dout, Hout, Wout);
        y[out_base] = acc0;
        if (v1) y[out_base + spatial] = acc1;
    }
}

// ------------------------ Fast: K=3x5x5, co2, W4 contiguous warp mapping, structured indexing ------------------------
// Thread linear maps to (n, co_pair, od, oh, ow4) with ow = ow4*4 + lane.
// This preserves contiguous ow across threads (unlike strided mappings) and improves coalescing.
template<bool USE_CONST_W>
__global__ __launch_bounds__(128, 3)
void conv_t3d_k3k5k5_s1p0_co2_w4(
    const float* __restrict__ x,   // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w,   // [Cin,Cout,3,5,5] (ignored if USE_CONST_W)
    float* __restrict__ y,         // [N,Cout,Dout,Hout,Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Dout, int Hout, int Wout
) {
    constexpr int Kd = 3, Kh = 5, Kw = 5;

    const int64_t spatial = (int64_t)Dout * Hout * Wout;
    const int co_pairs = (Cout + 1) >> 1;

    const int Wout4 = (Wout + 3) >> 2; // number of ow4 groups
    const int64_t total = (int64_t)N * co_pairs * (int64_t)Dout * (int64_t)Hout * (int64_t)Wout4;

    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = linear;

        int ow4 = (int)(t % Wout4); t /= Wout4;
        int oh  = (int)(t % Hout);  t /= Hout;
        int od  = (int)(t % Dout);  t /= Dout;
        int co_pair = (int)(t % co_pairs); t /= co_pairs;
        int n = (int)t;

        int co0 = co_pair * 2;
        int co1 = co0 + 1;
        bool v1 = (co1 < Cout);

        int ow0 = (ow4 << 2);

        // accumulators: [co2][w4]
        float acc00 = 0.f, acc01 = 0.f, acc02 = 0.f, acc03 = 0.f;
        float acc10 = 0.f, acc11 = 0.f, acc12 = 0.f, acc13 = 0.f;

        // bounds for kd/kh for this output (kw handled via predication per lane)
        int kd_min = od - (Din - 1); if (kd_min < 0) kd_min = 0;
        int kd_max = od; if (kd_max > 2) kd_max = 2;

        int kh_min = oh - (Hin - 1); if (kh_min < 0) kh_min = 0;
        int kh_max = oh; if (kh_max > 4) kh_max = 4;

        // Base pointers
        const int64_t x_hw = (int64_t)Hin * Win;
        const int64_t x_dhw = (int64_t)Din * x_hw;
        const int64_t x_cdhw = (int64_t)Cin * x_dhw;
        const int64_t x_n_base = (int64_t)n * x_cdhw;

        // weight base indices (for const path we compute linear index ourselves)
        const int64_t w_ci_stride = (int64_t)Cout * (Kd * Kh * Kw);
        const int64_t w_co_stride = (int64_t)(Kd * Kh * Kw);

        #pragma unroll 1
        for (int ci = 0; ci < Cin; ++ci) {
            const int64_t x_ci_base = x_n_base + (int64_t)ci * x_dhw;

            const float* __restrict__ w0_base_g = w + (int64_t)ci * w_ci_stride + (int64_t)co0 * w_co_stride;
            const float* __restrict__ w1_base_g = v1 ? (w + (int64_t)ci * w_ci_stride + (int64_t)co1 * w_co_stride) : nullptr;

            #pragma unroll 1
            for (int kd = kd_min; kd <= kd_max; ++kd) {
                int id = od - kd;
                const int64_t x_d_base = x_ci_base + (int64_t)id * x_hw;

                #pragma unroll 1
                for (int kh = kh_min; kh <= kh_max; ++kh) {
                    int ih = oh - kh;
                    const int64_t x_h_base = x_d_base + (int64_t)ih * Win;

                    // For W4, each lane has iw = (ow0 + lane) - kw. We predicate per lane/kw.
                    // Unroll kw=0..4 with scalar loads; compiler can schedule well.
                    #pragma unroll
                    for (int kw = 0; kw < 5; ++kw) {
                        int iw0 = ow0 - kw;
                        int iw1 = iw0 + 1;
                        int iw2 = iw0 + 2;
                        int iw3 = iw0 + 3;

                        bool p0 = (unsigned)ih < (unsigned)Hin && (unsigned)iw0 < (unsigned)Win && (ow0 + 0) < Wout;
                        bool p1 = (unsigned)ih < (unsigned)Hin && (unsigned)iw1 < (unsigned)Win && (ow0 + 1) < Wout;
                        bool p2 = (unsigned)ih < (unsigned)Hin && (unsigned)iw2 < (unsigned)Win && (ow0 + 2) < Wout;
                        bool p3 = (unsigned)ih < (unsigned)Hin && (unsigned)iw3 < (unsigned)Win && (ow0 + 3) < Wout;

                        float xv0 = p0 ? LDG(x + x_h_base + iw0) : 0.f;
                        float xv1 = p1 ? LDG(x + x_h_base + iw1) : 0.f;
                        float xv2 = p2 ? LDG(x + x_h_base + iw2) : 0.f;
                        float xv3 = p3 ? LDG(x + x_h_base + iw3) : 0.f;

                        int w_off = (kd * 25) + (kh * 5) + kw;

                        float w00, w10;
                        if constexpr (USE_CONST_W) {
                            // const layout matches global: [ci][co][kd][kh][kw]
                            int idx0 = (int)((((ci * Cout + co0) * 75) + w_off));
                            w00 = LDW_CONST(idx0);
                            if (v1) {
                                int idx1 = (int)((((ci * Cout + co1) * 75) + w_off));
                                w10 = LDW_CONST(idx1);
                            } else {
                                w10 = 0.f;
                            }
                        } else {
                            w00 = LDG(w0_base_g + w_off);
                            w10 = v1 ? LDG(w1_base_g + w_off) : 0.f;
                        }

                        acc00 = fmaf(xv0, w00, acc00);
                        acc01 = fmaf(xv1, w00, acc01);
                        acc02 = fmaf(xv2, w00, acc02);
                        acc03 = fmaf(xv3, w00, acc03);

                        if (v1) {
                            acc10 = fmaf(xv0, w10, acc10);
                            acc11 = fmaf(xv1, w10, acc11);
                            acc12 = fmaf(xv2, w10, acc12);
                            acc13 = fmaf(xv3, w10, acc13);
                        }
                    }
                }
            }
        }

        // store
        const int64_t y_hw = (int64_t)Hout * Wout;
        const int64_t y_dhw = (int64_t)Dout * y_hw;
        const int64_t y_n_base = (int64_t)n * (int64_t)Cout * y_dhw;
        const int64_t y_co0_base = y_n_base + (int64_t)co0 * y_dhw + (int64_t)od * y_hw + (int64_t)oh * Wout + ow0;

        if (ow0 + 0 < Wout) y[y_co0_base + 0] = acc00;
        if (ow0 + 1 < Wout) y[y_co0_base + 1] = acc01;
        if (ow0 + 2 < Wout) y[y_co0_base + 2] = acc02;
        if (ow0 + 3 < Wout) y[y_co0_base + 3] = acc03;

        if (v1) {
            const int64_t y_co1_base = y_co0_base + y_dhw;
            if (ow0 + 0 < Wout) y[y_co1_base + 0] = acc10;
            if (ow0 + 1 < Wout) y[y_co1_base + 1] = acc11;
            if (ow0 + 2 < Wout) y[y_co1_base + 2] = acc12;
            if (ow0 + 3 < Wout) y[y_co1_base + 3] = acc13;
        }
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor x, torch::Tensor w) {
    CHECK_CUDA(x);
    CHECK_CUDA(w);
    CHECK_FLOAT(x);
    CHECK_FLOAT(w);

    TORCH_CHECK(x.dim() == 5, "x must be 5D NCDHW");
    TORCH_CHECK(w.dim() == 5, "w must be 5D [Cin, Cout, Kd, Kh, Kw]");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    int64_t N   = x_c.size(0);
    int64_t Cin = x_c.size(1);
    int64_t Din = x_c.size(2);
    int64_t Hin = x_c.size(3);
    int64_t Win = x_c.size(4);

    TORCH_CHECK(w_c.size(0) == Cin, "w.size(0) must match Cin");
    int64_t Cout = w_c.size(1);
    int64_t Kd   = w_c.size(2);
    int64_t Kh   = w_c.size(3);
    int64_t Kw   = w_c.size(4);

    TORCH_CHECK(Hin == Win, "Expected square spatial input: Hin == Win");

    int64_t Dout = Din + Kd - 1;
    int64_t Hout = Hin + Kh - 1;
    int64_t Wout = Win + Kw - 1;

    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x_c.options());

    int device = x_c.get_device();
    cudaDeviceProp prop;
    int sm_count = 80;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) sm_count = prop.multiProcessorCount;

    const int threads = 128;

    // Prefer W4 fast path only for the benchmark kernel
    const bool fast_k = (Kd == 3 && Kh == 5 && Kw == 5);

    if (fast_k) {
        const int Wout4 = (int)((Wout + 3) >> 2);
        const int co_pairs = (int)((Cout + 1) >> 1);
        const int64_t total = (int64_t)N * co_pairs * Dout * Hout * Wout4;

        int blocks = (int)((total + threads - 1) / threads);
        int max_blocks = sm_count * 24;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;

        // Constant-memory path if the entire weight tensor fits
        const int64_t w_elems = Cin * Cout * 75;
        const bool can_const = (w_elems <= CONST_W_MAX);

        if (can_const) {
            // copy to constant (synchronous with respect to host, but small when it fits)
            cudaMemcpyToSymbol(const_w, w_c.data_ptr<float>(), (size_t)w_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);
            conv_t3d_k3k5k5_s1p0_co2_w4<true><<<blocks, threads>>>(
                x_c.data_ptr<float>(),
                w_c.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
                (int)Cout, (int)Dout, (int)Hout, (int)Wout
            );
        } else {
            conv_t3d_k3k5k5_s1p0_co2_w4<false><<<blocks, threads>>>(
                x_c.data_ptr<float>(),
                w_c.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
                (int)Cout, (int)Dout, (int)Hout, (int)Wout
            );
        }
        return y;
    }

    // Fallback: baseline generic co2 kernel
    const int64_t spatial = Dout * Hout * Wout;
    const int64_t total = N * ((Cout + 1) / 2) * spatial;

    int blocks = (int)((total + threads - 1) / threads);
    int max_blocks = sm_count * 24;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    conv_t3d_s1p0_generic_co2<<<blocks, threads>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
        (int)Cout, (int)Kd, (int)Kh, (int)Kw,
        (int)Dout, (int)Hout, (int)Wout
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose3d_cuda(torch::Tensor x, torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_square_input_asymmetric_v6",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        # keep occupancy healthy; avoid the known regression from raising this too high
        "-maxrregcount=80",
    ],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement for nn.ConvTranspose3d (bias=False, groups=1, stride=1, padding=0, output_padding=0)
    using an optimized custom CUDA kernel (forward only).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        kd, kh, kw = kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kd, kh, kw, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.output_padding = tuple(output_padding)
        self.groups = int(groups)
        self.bias_enabled = bool(bias)

        if self.stride != (1, 1, 1):
            raise ValueError("Custom kernel supports only stride=(1,1,1)")
        if self.padding != (0, 0, 0):
            raise ValueError("Custom kernel supports only padding=(0,0,0)")
        if self.output_padding != (0, 0, 0):
            raise ValueError("Custom kernel supports only output_padding=(0,0,0)")
        if self.groups != 1:
            raise ValueError("Custom kernel supports only groups=1")
        if self.bias_enabled:
            raise ValueError("Custom kernel supports only bias=False")

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops.conv_transpose3d_cuda(x, self.weight)