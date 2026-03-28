import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float LDG(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float LDG(const float* p) { return *p; }
#endif

// ---------------- Generic fallback (kept) ----------------
static __device__ __forceinline__ int64_t idx5d_ncdhw(
    int n, int c, int d, int h, int w,
    int C, int D, int H, int W
) {
    return (((((int64_t)n * C + c) * D + d) * H + h) * W + w);
}
static __device__ __forceinline__ int64_t idx5d_w(
    int ci, int co_g, int kd, int kh, int kw,
    int C_out_g, int Kd, int Kh, int Kw
) {
    return (((((int64_t)ci * C_out_g + co_g) * Kd + kd) * Kh + kh) * Kw + kw);
}

__global__ __launch_bounds__(128, 3)
void conv_t3d_generic_co2(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int Kd, int Kh, int Kw,
    int sd, int sh, int sw,
    int pd, int ph, int pw,
    int groups,
    bool has_bias
) {
    const int C_out_g = C_out / groups;
    const int C_in_g  = C_in  / groups;

    const int64_t spatial = (int64_t)D_out * H_out * W_out;
    const int64_t total_pairs = (int64_t)N * (int64_t)((C_out + 1) / 2) * spatial;

    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_pairs;
         linear += (int64_t)blockDim.x * gridDim.x) {

        int64_t t = linear;
        int ow = (int)(t % W_out); t /= W_out;
        int oh = (int)(t % H_out); t /= H_out;
        int od = (int)(t % D_out); t /= D_out;
        int co_pair = (int)(t % ((C_out + 1) / 2)); t /= ((C_out + 1) / 2);
        int n = (int)t;

        int co0 = co_pair * 2;
        int co1 = co0 + 1;
        bool valid1 = (co1 < C_out);

        float acc0 = 0.0f, acc1 = 0.0f;
        if (has_bias) {
            acc0 = LDG(b + co0);
            if (valid1) acc1 = LDG(b + co1);
        }

        int kd_min = max(0, od + pd - (D_in - 1) * sd);
        int kd_max = min(Kd - 1, od + pd);
        int kh_min = max(0, oh + ph - (H_in - 1) * sh);
        int kh_max = min(Kh - 1, oh + ph);
        int kw_min = max(0, ow + pw - (W_in - 1) * sw);
        int kw_max = min(Kw - 1, ow + pw);

        int g0 = co0 / C_out_g;
        int co0_g = co0 - g0 * C_out_g;

        int g1 = 0, co1_g = 0;
        if (valid1) {
            g1 = co1 / C_out_g;
            co1_g = co1 - g1 * C_out_g;
        }

        if (!valid1 || g0 == g1) {
            int ci_start = g0 * C_in_g;
            int ci_end = ci_start + C_in_g;
            for (int ci = ci_start; ci < ci_end; ++ci) {
                for (int kd = kd_min; kd <= kd_max; ++kd) {
                    int num_d = od + pd - kd;
                    if (num_d % sd) continue;
                    int id = num_d / sd;
                    if ((unsigned)id >= (unsigned)D_in) continue;

                    for (int kh = kh_min; kh <= kh_max; ++kh) {
                        int num_h = oh + ph - kh;
                        if (num_h % sh) continue;
                        int ih = num_h / sh;
                        if ((unsigned)ih >= (unsigned)H_in) continue;

                        for (int kw = kw_min; kw <= kw_max; ++kw) {
                            int num_w = ow + pw - kw;
                            if (num_w % sw) continue;
                            int iw = num_w / sw;
                            if ((unsigned)iw >= (unsigned)W_in) continue;

                            float xv = LDG(x + idx5d_ncdhw(n, ci, id, ih, iw, C_in, D_in, H_in, W_in));
                            acc0 = fmaf(xv, LDG(w + idx5d_w(ci, co0_g, kd, kh, kw, C_out_g, Kd, Kh, Kw)), acc0);
                            if (valid1) acc1 = fmaf(xv, LDG(w + idx5d_w(ci, co1_g, kd, kh, kw, C_out_g, Kd, Kh, Kw)), acc1);
                        }
                    }
                }
            }
        } else {
            {
                int ci_start = g0 * C_in_g;
                int ci_end = ci_start + C_in_g;
                for (int ci = ci_start; ci < ci_end; ++ci) {
                    for (int kd = kd_min; kd <= kd_max; ++kd) {
                        int num_d = od + pd - kd;
                        if (num_d % sd) continue;
                        int id = num_d / sd;
                        if ((unsigned)id >= (unsigned)D_in) continue;
                        for (int kh = kh_min; kh <= kh_max; ++kh) {
                            int num_h = oh + ph - kh;
                            if (num_h % sh) continue;
                            int ih = num_h / sh;
                            if ((unsigned)ih >= (unsigned)H_in) continue;
                            for (int kw = kw_min; kw <= kw_max; ++kw) {
                                int num_w = ow + pw - kw;
                                if (num_w % sw) continue;
                                int iw = num_w / sw;
                                if ((unsigned)iw >= (unsigned)W_in) continue;
                                float xv = LDG(x + idx5d_ncdhw(n, ci, id, ih, iw, C_in, D_in, H_in, W_in));
                                acc0 = fmaf(xv, LDG(w + idx5d_w(ci, co0_g, kd, kh, kw, C_out_g, Kd, Kh, Kw)), acc0);
                            }
                        }
                    }
                }
            }
            if (valid1) {
                int ci_start = g1 * C_in_g;
                int ci_end = ci_start + C_in_g;
                for (int ci = ci_start; ci < ci_end; ++ci) {
                    for (int kd = kd_min; kd <= kd_max; ++kd) {
                        int num_d = od + pd - kd;
                        if (num_d % sd) continue;
                        int id = num_d / sd;
                        if ((unsigned)id >= (unsigned)D_in) continue;
                        for (int kh = kh_min; kh <= kh_max; ++kh) {
                            int num_h = oh + ph - kh;
                            if (num_h % sh) continue;
                            int ih = num_h / sh;
                            if ((unsigned)ih >= (unsigned)H_in) continue;
                            for (int kw = kw_min; kw <= kw_max; ++kw) {
                                int num_w = ow + pw - kw;
                                if (num_w % sw) continue;
                                int iw = num_w / sw;
                                if ((unsigned)iw >= (unsigned)W_in) continue;
                                float xv = LDG(x + idx5d_ncdhw(n, ci, id, ih, iw, C_in, D_in, H_in, W_in));
                                acc1 = fmaf(xv, LDG(w + idx5d_w(ci, co1_g, kd, kh, kw, C_out_g, Kd, Kh, Kw)), acc1);
                            }
                        }
                    }
                }
            }
        }

        int64_t base = ((((int64_t)n * C_out + co0) * D_out + od) * H_out + oh) * W_out + ow;
        out[base] = acc0;
        if (valid1) out[base + spatial] = acc1;
    }
}

// ---------------- Fast path: groups=1, stride=1, padding=0, output_padding=0, K=3x5x7, Cout even ----------------
// Correct transposed-conv mapping for stride=1,pad=0: id = od - kd, etc., with bounds checks.
//
// Strategy: 256-thread CTA, each thread computes 2 contiguous flattened spatial positions.
// This reduces per-thread work vs previous 4 outputs/thread and increases active warps for latency hiding.
// We keep weight loads scalar (no unsafe float2 casting).
__global__ __launch_bounds__(256, 2)
void convt3d_k3k5k7_s1_p0_g1_pairs_tile2_boundary_correct(
    const float* __restrict__ x, // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w, // [Cin,Cout,3,5,7] (since groups=1, Cout_g=Cout)
    const float* __restrict__ b, // [Cout] or nullptr
    float* __restrict__ y,       // [N,Cout,Dout,Hout,Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Dout, int Hout, int Wout,
    bool has_bias
) {
    constexpr int Kd = 3, Kh = 5, Kw = 7;

    const int co_pairs = (Cout >> 1);
    const int by = (int)blockIdx.y;
    const int n = by / co_pairs;
    const int co_pair = by - n * co_pairs;
    if ((unsigned)n >= (unsigned)N) return;

    const int co0 = co_pair * 2;
    const int co1 = co0 + 1;

    const int64_t spatial = (int64_t)Dout * Hout * Wout;
    const int64_t tile_base = (int64_t)blockIdx.x * (int64_t)blockDim.x * 2;

    const int64_t out_base_n = (int64_t)n * (int64_t)Cout * spatial;
    const int64_t out_base0 = out_base_n + (int64_t)co0 * spatial;
    const int64_t out_base1 = out_base0 + spatial;

    const int64_t x_n_base = (int64_t)n * (int64_t)Cin * (int64_t)Din * Hin * Win;

    float bias0 = 0.f, bias1 = 0.f;
    if (has_bias) {
        bias0 = LDG(b + co0);
        bias1 = LDG(b + co1);
    }

    // Compute two consecutive flattened indices per thread: s0 and s1 = s0 + blockDim.x
    int64_t s0 = tile_base + (int64_t)threadIdx.x;
    if (s0 >= spatial) return;

    // Decode s0 -> (od,oh,ow)
    int64_t t0 = s0;
    int ow0 = (int)(t0 % Wout); t0 /= Wout;
    int oh0 = (int)(t0 % Hout); t0 /= Hout;
    int od0 = (int)t0;

    // s1, update coordinates incrementally to avoid extra div/mod
    int64_t s1 = s0 + (int64_t)blockDim.x;
    int od1 = od0, oh1 = oh0, ow1 = ow0;
    if (s1 < spatial) {
        int ow_tmp = ow0 + (int)blockDim.x;
        int carry_h = ow_tmp / Wout;
        ow1 = ow_tmp - carry_h * Wout;
        int oh_tmp = oh0 + carry_h;
        int carry_d = oh_tmp / Hout;
        oh1 = oh_tmp - carry_d * Hout;
        od1 = od0 + carry_d;
    }

    // Lambda-like inline accumulation for one output point
    auto compute_one = [&](int od, int oh, int ow, float &acc0, float &acc1) {
        acc0 = has_bias ? bias0 : 0.f;
        acc1 = has_bias ? bias1 : 0.f;

        // Bounds of kd/kh/kw such that id/ih/iw stay in-range for stride=1,pad=0
        // id = od - kd in [0, Din-1] => kd in [od-(Din-1), od]
        int kd_min = od - (Din - 1); if (kd_min < 0) kd_min = 0;
        int kd_max = od;            if (kd_max > (Kd - 1)) kd_max = (Kd - 1);

        int kh_min = oh - (Hin - 1); if (kh_min < 0) kh_min = 0;
        int kh_max = oh;             if (kh_max > (Kh - 1)) kh_max = (Kh - 1);

        int kw_min = ow - (Win - 1); if (kw_min < 0) kw_min = 0;
        int kw_max = ow;             if (kw_max > (Kw - 1)) kw_max = (Kw - 1);

        #pragma unroll 1
        for (int ci = 0; ci < Cin; ++ci) {
            const float* __restrict__ x_c = x + x_n_base + (int64_t)ci * (int64_t)Din * Hin * Win;
            const float* __restrict__ w0 = w + ((int64_t)ci * (int64_t)Cout + (int64_t)co0) * (Kd * Kh * Kw);
            const float* __restrict__ w1 = w + ((int64_t)ci * (int64_t)Cout + (int64_t)co1) * (Kd * Kh * Kw);

            #pragma unroll
            for (int kd = 0; kd < Kd; ++kd) {
                if (kd < kd_min || kd > kd_max) continue;
                const int id = od - kd;
                const float* __restrict__ x_d = x_c + (int64_t)id * Hin * Win;
                const float* __restrict__ w0_kd = w0 + kd * (Kh * Kw);
                const float* __restrict__ w1_kd = w1 + kd * (Kh * Kw);

                #pragma unroll
                for (int kh = 0; kh < Kh; ++kh) {
                    if (kh < kh_min || kh > kh_max) continue;
                    const int ih = oh - kh;
                    const float* __restrict__ x_dh = x_d + (int64_t)ih * Win;
                    const float* __restrict__ w0_kh = w0_kd + kh * Kw;
                    const float* __restrict__ w1_kh = w1_kd + kh * Kw;

                    float xv;
                    if (kw_min <= 0 && 0 <= kw_max) { xv = LDG(x_dh + (ow - 0)); acc0 = fmaf(xv, LDG(w0_kh + 0), acc0); acc1 = fmaf(xv, LDG(w1_kh + 0), acc1); }
                    if (kw_min <= 1 && 1 <= kw_max) { xv = LDG(x_dh + (ow - 1)); acc0 = fmaf(xv, LDG(w0_kh + 1), acc0); acc1 = fmaf(xv, LDG(w1_kh + 1), acc1); }
                    if (kw_min <= 2 && 2 <= kw_max) { xv = LDG(x_dh + (ow - 2)); acc0 = fmaf(xv, LDG(w0_kh + 2), acc0); acc1 = fmaf(xv, LDG(w1_kh + 2), acc1); }
                    if (kw_min <= 3 && 3 <= kw_max) { xv = LDG(x_dh + (ow - 3)); acc0 = fmaf(xv, LDG(w0_kh + 3), acc0); acc1 = fmaf(xv, LDG(w1_kh + 3), acc1); }
                    if (kw_min <= 4 && 4 <= kw_max) { xv = LDG(x_dh + (ow - 4)); acc0 = fmaf(xv, LDG(w0_kh + 4), acc0); acc1 = fmaf(xv, LDG(w1_kh + 4), acc1); }
                    if (kw_min <= 5 && 5 <= kw_max) { xv = LDG(x_dh + (ow - 5)); acc0 = fmaf(xv, LDG(w0_kh + 5), acc0); acc1 = fmaf(xv, LDG(w1_kh + 5), acc1); }
                    if (kw_min <= 6 && 6 <= kw_max) { xv = LDG(x_dh + (ow - 6)); acc0 = fmaf(xv, LDG(w0_kh + 6), acc0); acc1 = fmaf(xv, LDG(w1_kh + 6), acc1); }
                }
            }
        }
    };

    float a00, a01;
    compute_one(od0, oh0, ow0, a00, a01);
    y[out_base0 + s0] = a00;
    y[out_base1 + s0] = a01;

    if (s1 < spatial) {
        float a10, a11;
        compute_one(od1, oh1, ow1, a10, a11);
        y[out_base0 + s1] = a10;
        y[out_base1 + s1] = a11;
    }
}

torch::Tensor conv_transposed3d_asymmetric_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C_in,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be 5D [C_in, C_out/groups, Kd,Kh,Kw]");
    TORCH_CHECK(stride.size() == 3 && padding.size() == 3 && output_padding.size() == 3, "stride/padding/output_padding must be len=3");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int64_t N = x_c.size(0);
    const int64_t C_in = x_c.size(1);
    const int64_t D_in = x_c.size(2);
    const int64_t H_in = x_c.size(3);
    const int64_t W_in = x_c.size(4);

    const int64_t Kd = w_c.size(2);
    const int64_t Kh = w_c.size(3);
    const int64_t Kw = w_c.size(4);

    TORCH_CHECK(w_c.size(0) == C_in, "w.shape[0] must equal C_in");
    TORCH_CHECK(C_in % groups == 0, "C_in must be divisible by groups");

    const int64_t C_out_g = w_c.size(1);
    const int64_t C_out = C_out_g * groups;

    const int sd = (int)stride[0], sh = (int)stride[1], sw = (int)stride[2];
    const int pd = (int)padding[0], ph = (int)padding[1], pw = (int)padding[2];
    const int opd = (int)output_padding[0], oph = (int)output_padding[1], opw = (int)output_padding[2];

    TORCH_CHECK(sd >= 1 && sh >= 1 && sw >= 1, "stride must be >= 1");
    TORCH_CHECK(pd >= 0 && ph >= 0 && pw >= 0, "padding must be >= 0");
    TORCH_CHECK(opd >= 0 && oph >= 0 && opw >= 0, "output_padding must be >= 0");

    const int64_t D_out = (D_in - 1) * sd - 2 * pd + Kd + opd;
    const int64_t H_out = (H_in - 1) * sh - 2 * ph + Kh + oph;
    const int64_t W_out = (W_in - 1) * sw - 2 * pw + Kw + opw;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "computed output size must be positive");

    torch::Tensor b;
    bool has_bias = false;
    if (bias_opt.has_value() && bias_opt.value().defined() && bias_opt.value().numel() > 0) {
        b = bias_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == C_out, "bias must be [C_out]");
        b = b.contiguous();
        has_bias = true;
    }
    const float* b_ptr = has_bias ? b.data_ptr<float>() : nullptr;

    auto out = torch::empty({N, C_out, D_out, H_out, W_out}, x_c.options());

    const bool fast_path =
        (groups == 1) &&
        (Kd == 3 && Kh == 5 && Kw == 7) &&
        (sd == 1 && sh == 1 && sw == 1) &&
        (pd == 0 && ph == 0 && pw == 0) &&
        (opd == 0 && oph == 0 && opw == 0) &&
        ((C_out & 1) == 0);

    int device = x_c.get_device();
    cudaDeviceProp prop;
    int sm_count = 80;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) sm_count = prop.multiProcessorCount;

    if (fast_path) {
        const int threads = 256;
        const int64_t spatial = D_out * H_out * W_out;
        const int64_t tile_elems = (int64_t)threads * 2;

        int grid_x = (int)((spatial + tile_elems - 1) / tile_elems);
        int max_grid_x = sm_count * 12;
        if (grid_x > max_grid_x) grid_x = max_grid_x;
        if (grid_x < 1) grid_x = 1;

        dim3 grid((unsigned)grid_x, (unsigned)(N * (C_out / 2)), 1);
        convt3d_k3k5k7_s1_p0_g1_pairs_tile2_boundary_correct<<<grid, threads>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            out.data_ptr<float>(),
            (int)N, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
            (int)C_out, (int)D_out, (int)H_out, (int)W_out,
            has_bias
        );
    } else {
        const int threads = 128;
        const int64_t spatial = D_out * H_out * W_out;
        const int64_t total_pairs = N * ((C_out + 1) / 2) * spatial;
        int blocks = (int)((total_pairs + threads - 1) / threads);
        int max_blocks = sm_count * 32;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;

        conv_t3d_generic_co2<<<blocks, threads>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            out.data_ptr<float>(),
            (int)N, (int)C_in, (int)D_in, (int)H_in, (int)W_in,
            (int)C_out, (int)D_out, (int)H_out, (int)W_out,
            (int)Kd, (int)Kh, (int)Kw,
            sd, sh, sw,
            pd, ph, pw,
            (int)groups,
            has_bias
        );
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transposed3d_asymmetric_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_asym_opt10_tile2_single_fast",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transposed3d_asymmetric_input_asymmetric_kernel_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        # Encourage lower register count to raise occupancy; avoid being too tight to prevent spills.
        "-maxrregcount=64",
    ],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement for nn.ConvTranspose3d using a custom CUDA kernel.
    Keeps a real ConvTranspose3d module only as a parameter container.
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
        self.custom_ops = custom_ops_lib
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self._stride = tuple(int(v) for v in self.conv_transpose3d.stride)
        self._padding = tuple(int(v) for v in self.conv_transpose3d.padding)
        self._output_padding = tuple(int(v) for v in self.conv_transpose3d.output_padding)
        self._groups = int(self.conv_transpose3d.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        return self.custom_ops.conv_transposed3d_asymmetric_input_asymmetric_kernel_cuda(
            x, w, b, list(self._stride), list(self._padding), list(self._output_padding), self._groups
        )