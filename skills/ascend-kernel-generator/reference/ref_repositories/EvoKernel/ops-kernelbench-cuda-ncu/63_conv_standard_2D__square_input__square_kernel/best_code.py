import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif

// We support float32 only in this optimized path.
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

// Specialization matches the provided reference conv: Cin=16, K=3, stride=1, pad=0, dilation=1, groups=1, bias optional.
// Square input/kernel is implied by sizes.
constexpr int CIN = 16;
constexpr int K = 3;
constexpr int STRIDE = 1;
constexpr int PAD = 0;

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

// Border kernel: computes only outside interior box [oh0,oh1) x [ow0,ow1).
__global__ void conv2d_fwd_nchw_k3s1p0_border_only(
    const float* __restrict__ x,  // [N,CIN,Hin,Win]
    const float* __restrict__ w,  // [Cout,CIN,3,3]
    const float* __restrict__ b,  // [Cout] or nullptr
    float* __restrict__ y,        // [N,Cout,Hout,Wout]
    int N, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int oh0, int oh1, int ow0, int ow1
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    // Skip interior
    if ((unsigned)(oh - oh0) < (unsigned)(oh1 - oh0) &&
        (unsigned)(ow - ow0) < (unsigned)(ow1 - ow0)) {
        return;
    }

    float acc = 0.0f;
    if (b) acc = ldg_f32(b + oc);

    int in_y0 = oh * STRIDE - PAD;
    int in_x0 = ow * STRIDE - PAD;

    int w_oc_base = oc * (CIN * K * K);

    // For border, need bounds checks.
    for (int ic = 0; ic < CIN; ++ic) {
        int x_ic_base = ((n * CIN + ic) * Hin) * Win;
        int w_ic_base = w_oc_base + ic * (K * K);

        for (int ky = 0; ky < K; ++ky) {
            int iy = in_y0 + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            int x_row = x_ic_base + iy * Win;
            int w_row = w_ic_base + ky * K;

            for (int kx = 0; kx < K; ++kx) {
                int ix = in_x0 + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;
                float xv = ldg_f32(x + (x_row + ix));
                float wv = ldg_f32(w + (w_row + kx));
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

// Interior kernel: fixed (n,oc) per blockIdx.y. Each thread computes 2 output pixels in ow.
// No shared memory / no barrier to reduce sync stalls; we rely on read-only cache for weights.
__global__ __launch_bounds__(128, 4) void conv2d_fwd_nchw_k3s1p0_interior_ocblock_ow2_nosmem(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int oh0, int oh1, int ow0, int ow1_even
) {
    int noc = (int)blockIdx.y;
    int n = noc / Cout;
    int oc = noc - n * Cout;

    int interior_h = oh1 - oh0;
    int interior_w_even = ow1_even - ow0;
    if (interior_h <= 0 || interior_w_even < 2) return;

    int owPairs = interior_w_even >> 1;
    int numTiles = interior_h * owPairs;

    float bval = 0.0f;
    if (b) bval = ldg_f32(b + oc);

    int w_oc_base = oc * (CIN * K * K);

    // grid-stride over tiles; tile index = (oh-oh0)*owPairs + owPair
    for (int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         t < numTiles;
         t += (int)gridDim.x * (int)blockDim.x) {

        int owPair = t % owPairs;
        int oh = oh0 + (t / owPairs);

        int owA = ow0 + (owPair << 1);
        int owB = owA + 1;

        float accA = bval;
        float accB = bval;

        int in_y0 = oh;              // stride=1, pad=0
        int in_x0A = owA;
        int in_x0B = owB;

        // Interior guarantees in_y0..in_y0+2 in-bounds and in_x0..in_x0+2 in-bounds.
        // Keep loops not fully unrolled to reduce register pressure.
        for (int ic = 0; ic < CIN; ++ic) {
            int x_ic_base = ((n * CIN + ic) * Hin + in_y0) * Win;
            int w_ic_base = w_oc_base + ic * (K * K);

            // ky=0..2
            #pragma unroll 1
            for (int ky = 0; ky < K; ++ky) {
                int x_row = x_ic_base + ky * Win;
                int w_row = w_ic_base + ky * K;

                // kx=0..2
                #pragma unroll 1
                for (int kx = 0; kx < K; ++kx) {
                    float wv = ldg_f32(w + (w_row + kx));
                    float xvA = ldg_f32(x + (x_row + in_x0A + kx));
                    float xvB = ldg_f32(x + (x_row + in_x0B + kx));
                    accA = fmaf(xvA, wv, accA);
                    accB = fmaf(xvB, wv, accB);
                }
            }
        }

        int out_base = ((n * Cout + oc) * Hout + oh) * Wout;
        y[out_base + owA] = accA;
        y[out_base + owB] = accB;
    }
}

// Leftover interior columns kernel: compute only ow in [ow1_even, ow1) for oh in [oh0,oh1) (typically 0 or 1 column).
__global__ __launch_bounds__(256, 2) void conv2d_fwd_nchw_k3s1p0_leftover_cols(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int oh0, int oh1, int ow_start, int ow_end
) {
    int interior_h = oh1 - oh0;
    int interior_w = ow_end - ow_start;
    if (interior_h <= 0 || interior_w <= 0) return;

    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = N * Cout * interior_h * interior_w;
    if (idx >= total) return;

    int ow_local = idx % interior_w;
    int t1 = idx / interior_w;
    int oh_local = t1 % interior_h;
    int t2 = t1 / interior_h;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int ow = ow_start + ow_local;
    int oh = oh0 + oh_local;

    float acc = 0.0f;
    if (b) acc = ldg_f32(b + oc);

    int w_oc_base = oc * (CIN * K * K);
    int in_y0 = oh;
    int in_x0 = ow;

    // interior => no bounds checks
    for (int ic = 0; ic < CIN; ++ic) {
        int x_ic_base = ((n * CIN + ic) * Hin + in_y0) * Win;
        int w_ic_base = w_oc_base + ic * (K * K);

        #pragma unroll 1
        for (int ky = 0; ky < K; ++ky) {
            int x_row = x_ic_base + ky * Win;
            int w_row = w_ic_base + ky * K;

            #pragma unroll 1
            for (int kx = 0; kx < K; ++kx) {
                float xv = ldg_f32(x + (x_row + in_x0 + kx));
                float wv = ldg_f32(w + (w_row + kx));
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

torch::Tensor conv2d_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    CHECK_CUDA(x); CHECK_CUDA(w);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w);
    CHECK_FLOAT(x); CHECK_FLOAT(w);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW");
    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);

    int Cout = (int)w.size(0);

    TORCH_CHECK(Cin == CIN, "Specialized kernel expects in_channels=16");
    TORCH_CHECK((int)w.size(1) == Cin, "weight Cin mismatch");
    TORCH_CHECK((int)w.size(2) == K && (int)w.size(3) == K, "weight must be 3x3");

    // stride=1, pad=0
    int Hout = (Hin - K) / STRIDE + 1;
    int Wout = (Win - K) / STRIDE + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    const float* bptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        CHECK_CUDA(b);
        CHECK_CONTIGUOUS(b);
        CHECK_FLOAT(b);
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == Cout, "bias must be [Cout]");
        bptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    // interior where full receptive field is in bounds. For pad=0,stride=1,k=3 => oh in [0,Hout), ow in [0,Wout) is all interior.
    int oh0 = 0, oh1 = Hout;
    int ow0 = 0, ow1 = Wout;

    int interior_w = ow1 - ow0;
    int ow1_even = ow0 + (interior_w & ~1);

    // Interior ow2 kernel
    if ((oh1 - oh0) > 0 && (ow1_even - ow0) >= 2) {
        int owPairs = (ow1_even - ow0) >> 1;
        int numTiles = (oh1 - oh0) * owPairs;

        int block = 128;
        int grid_x = (numTiles + block - 1) / block;
        if (grid_x < 80) grid_x = 80;
        if (grid_x > 4096) grid_x = 4096;

        dim3 grid(grid_x, N * Cout, 1);
        conv2d_fwd_nchw_k3s1p0_interior_ocblock_ow2_nosmem<<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bptr,
            y.data_ptr<float>(),
            N, Hin, Win,
            Cout, Hout, Wout,
            oh0, oh1, ow0, ow1_even
        );
    }

    // Leftover interior columns (typically 0 or 1)
    if (ow1_even < ow1) {
        int cols = ow1 - ow1_even;
        int interior_h = oh1 - oh0;
        int total = N * Cout * interior_h * cols;
        int block = 256;
        int grid = (total + block - 1) / block;
        conv2d_fwd_nchw_k3s1p0_leftover_cols<<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bptr,
            y.data_ptr<float>(),
            N, Hin, Win,
            Cout, Hout, Wout,
            oh0, oh1, ow1_even, ow1
        );
    }

    // True border: for pad=0 and full interior covering all outputs, border is empty.
    // Keep border kernel for safety/generalization if shapes ever change unexpectedly.
    // Here the interior box is full [0,Hout)x[0,Wout), so border kernel does nothing.
    {
        int block = 256;
        int total = N * Cout * Hout * Wout;
        int grid = (total + block - 1) / block;
        conv2d_fwd_nchw_k3s1p0_border_only<<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bptr,
            y.data_ptr<float>(),
            N, Hin, Win,
            Cout, Hout, Wout,
            oh0, oh1, ow0, ow1
        );
    }

    return y;
}
"""

conv2d_cpp_source = r"""
torch::Tensor conv2d_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_c16_k3s1p0_v1",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep semantics aligned with the reference operator; enforce only what the specialized kernel supports.
        if in_channels != 16 or kernel_size != 3 or stride != 1 or padding != 0 or dilation != 1 or groups != 1:
            raise ValueError("This optimized kernel supports in_channels=16, kernel_size=3, stride=1, padding=0, dilation=1, groups=1.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = bool(bias)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.use_bias:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        w = self.weight.contiguous()
        if self.bias is None:
            b = torch.empty((0,), device=x.device, dtype=x.dtype)
        else:
            b = self.bias.contiguous()
        return custom_ops_lib.conv2d_forward_cuda(x, w, b)