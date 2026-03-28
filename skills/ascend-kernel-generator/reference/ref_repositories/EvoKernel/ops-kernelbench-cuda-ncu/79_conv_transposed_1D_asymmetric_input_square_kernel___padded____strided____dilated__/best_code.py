import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline int div_up_ll(long long a, int b) { return (int)((a + b - 1) / b); }

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float ld_g(const float* p) { return __ldg(p); }
#else
__host__ __forceinline__ float ld_g(const float* p) { return *p; }
#endif

// ---------------- Generic fallback kernel ----------------
__global__ void conv_transpose1d_forward_generic(
    const float* __restrict__ x,      // [N, Cin, Lin]
    const float* __restrict__ w,      // [Cin, Cout, K]
    const float* __restrict__ b,      // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout, int K,
    int stride, int padding, int dilation,
    int Lout
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * (long long)Cout * (long long)Lout;
    if (idx >= total) return;

    int ox = (int)(idx % Lout);
    int oc = (int)((idx / Lout) % Cout);
    int n  = (int)(idx / ((long long)Lout * Cout));

    float acc = 0.0f;
    if (b) acc = ld_g(b + oc);

    const long long x_n_base = (long long)n * Cin * Lin;
    for (int ic = 0; ic < Cin; ++ic) {
        const float* x_ic = x + x_n_base + (long long)ic * Lin;
        const float* w_ic_oc = w + ((long long)ic * Cout + oc) * K;
        #pragma unroll 1
        for (int k = 0; k < K; ++k) {
            int t = ox + padding - k * dilation;
            if (t < 0) continue;
            if (t % stride != 0) continue;
            int ix = t / stride;
            if ((unsigned)ix >= (unsigned)Lin) continue;
            acc = fmaf(ld_g(x_ic + ix), ld_g(w_ic_oc + k), acc);
        }
    }
    y[((long long)n * Cout + oc) * Lout + ox] = acc;
}

// ---------------- Constant memory for specialized fixed shape ----------------
// Fixed specialized shape: Cin=32, Cout=64, K=3.
// Layout: ((ic * 64 + oc) * 3 + k)
__constant__ float c_w_32_64_3[32 * 64 * 3];
__constant__ float c_b_64[64];

static inline void cuda_check(cudaError_t e) {
    TORCH_CHECK(e == cudaSuccess, "CUDA error: ", cudaGetErrorString(e));
}

static void upload_const_wb(const float* w_host_or_dev, const float* b_host_or_dev, bool has_bias) {
    // Device-to-device copies are allowed for cudaMemcpyToSymbol with cudaMemcpyDeviceToDevice.
    cuda_check(cudaMemcpyToSymbol(c_w_32_64_3, w_host_or_dev, sizeof(float) * 32 * 64 * 3, 0, cudaMemcpyDeviceToDevice));
    if (has_bias) {
        cuda_check(cudaMemcpyToSymbol(c_b_64, b_host_or_dev, sizeof(float) * 64, 0, cudaMemcpyDeviceToDevice));
    }
}

// ---------------- Specialized kernels for K=3, stride=2, dilation=2, padding=1, Cin=32, Cout=64 ----------------
//
// Observation: Only odd ox have contributions.
// For odd ox: q = (ox+1)/2, ix = q - k (k=0,1,2). Need ix in [0, Lin).
// For even ox: output is bias (or 0) only.

__global__ __launch_bounds__(128, 3)
void conv_t1d_s32_c64_k3_s2_d2_p1_odd_oc2(
    const float* __restrict__ x,  // [N,32,Lin]
    float* __restrict__ y,        // [N,64,Lout]
    int N, int Lin, int Lout,
    bool has_bias
) {
    // Map threads to (n, oc_pair, ox_odd)
    // ox_odd enumerates odd positions: ox = 2*i + 1
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int odd_count = (Lout >> 1); // number of odd positions in [0..Lout-1]
    int oc_pairs = 32;           // 64/2
    int per_n = oc_pairs * odd_count;
    int total = N * per_n;
    if (tid >= total) return;

    int t = tid;
    int i_odd = t % odd_count; t /= odd_count;
    int oc_pair = t % oc_pairs; t /= oc_pairs;
    int n = t;

    int ox = (i_odd << 1) + 1;
    int oc0 = oc_pair << 1;
    int oc1 = oc0 + 1;

    float acc0 = has_bias ? c_b_64[oc0] : 0.0f;
    float acc1 = has_bias ? c_b_64[oc1] : 0.0f;

    int q = (ox + 1) >> 1; // integer

    long long x_n_base = (long long)n * 32 * Lin;

    #pragma unroll
    for (int ic = 0; ic < 32; ++ic) {
        const float* x_ic = x + x_n_base + (long long)ic * Lin;

        // weights for oc0 and oc1 from constant memory
        int wbase0 = (ic * 64 + oc0) * 3;
        int wbase1 = wbase0 + 3;

        float w00 = c_w_32_64_3[wbase0 + 0];
        float w01 = c_w_32_64_3[wbase0 + 1];
        float w02 = c_w_32_64_3[wbase0 + 2];
        float w10 = c_w_32_64_3[wbase1 + 0];
        float w11 = c_w_32_64_3[wbase1 + 1];
        float w12 = c_w_32_64_3[wbase1 + 2];

        int ix0 = q;
        int ix1 = q - 1;
        int ix2 = q - 2;

        if ((unsigned)ix0 < (unsigned)Lin) {
            float xv = ld_g(x_ic + ix0);
            acc0 = fmaf(xv, w00, acc0);
            acc1 = fmaf(xv, w10, acc1);
        }
        if ((unsigned)ix1 < (unsigned)Lin) {
            float xv = ld_g(x_ic + ix1);
            acc0 = fmaf(xv, w01, acc0);
            acc1 = fmaf(xv, w11, acc1);
        }
        if ((unsigned)ix2 < (unsigned)Lin) {
            float xv = ld_g(x_ic + ix2);
            acc0 = fmaf(xv, w02, acc0);
            acc1 = fmaf(xv, w12, acc1);
        }
    }

    long long y_off0 = ((long long)n * 64 + oc0) * Lout + ox;
    long long y_off1 = y_off0 + (long long)Lout; // next channel

    // Store two channels (non-contiguous), scalar stores
    y[y_off0] = acc0;
    y[y_off1] = acc1;
}

__global__ __launch_bounds__(256, 2)
void conv_t1d_s32_c64_k3_s2_d2_p1_even_bias_only(
    float* __restrict__ y, // [N,64,Lout]
    int N, int Lout,
    bool has_bias
) {
    // Fill even ox: ox = 2*i
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long even_count = (Lout + 1) >> 1; // number of even positions
    long long total = (long long)N * 64 * even_count;
    if (tid >= total) return;

    int i_even = (int)(tid % even_count);
    int oc = (int)((tid / even_count) % 64);
    int n = (int)(tid / (even_count * 64));

    int ox = i_even << 1;
    if (ox >= Lout) return;

    float v = has_bias ? c_b_64[oc] : 0.0f;
    y[((long long)n * 64 + oc) * Lout + ox] = v;
}

torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor x,          // [N, Cin, Lin]
    torch::Tensor w,          // [Cin, Cout, K]
    c10::optional<torch::Tensor> b_opt, // [Cout] optional
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 3, "x must be NCL (3D)");
    TORCH_CHECK(w.dim() == 3, "w must be (Cin, Cout, K) (3D)");

    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(dilation > 0, "dilation must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    const int N = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int Lin = (int)x.size(2);

    TORCH_CHECK((int)w.size(0) == Cin, "w.size(0) must equal Cin");
    const int Cout = (int)w.size(1);
    const int K = (int)w.size(2);
    TORCH_CHECK(K > 0, "kernel size must be > 0");

    const int Lout = (int)((Lin - 1) * (int)stride - 2 * (int)padding + (int)dilation * (K - 1) + 1);
    TORCH_CHECK(Lout > 0, "computed output length is non-positive");

    const bool has_bias = b_opt.has_value() && b_opt.value().defined();
    const float* b_ptr = nullptr;
    torch::Tensor b;
    if (has_bias) {
        b = b_opt.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == Cout, "bias must be 1D of size Cout");
        b_ptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, Lout}, x.options());

    const bool use_spec =
        (Cin == 32 && Cout == 64 && K == 3 &&
         (int)stride == 2 && (int)dilation == 2 && (int)padding == 1);

    if (use_spec) {
        // Upload weights/bias to constant memory (simple always-upload; safe and typically amortized in steady-state).
        upload_const_wb(w.data_ptr<float>(), has_bias ? b_ptr : nullptr, has_bias);

        // 1) Fill even positions (bias-only) - cheap
        {
            const int threads = 256;
            long long even_count = (Lout + 1) >> 1;
            long long total = (long long)N * 64 * even_count;
            int blocks = div_up_ll(total, threads);
            conv_t1d_s32_c64_k3_s2_d2_p1_even_bias_only<<<blocks, threads>>>(
                y.data_ptr<float>(), N, Lout, has_bias
            );
        }
        // 2) Compute odd positions with oc2 blocking
        {
            const int threads = 128;
            int odd_count = (Lout >> 1);
            int total = N * 32 * odd_count;
            int blocks = (total + threads - 1) / threads;
            conv_t1d_s32_c64_k3_s2_d2_p1_odd_oc2<<<blocks, threads>>>(
                x.data_ptr<float>(),
                y.data_ptr<float>(),
                N, Lin, Lout,
                has_bias
            );
        }
        return y;
    }

    // Fallback generic
    {
        const float* bptr = has_bias ? b_ptr : nullptr;
        const int threads = 256;
        long long total = (long long)N * (long long)Cout * (long long)Lout;
        const int blocks = div_up_ll(total, threads);
        conv_transpose1d_forward_generic<<<blocks, threads>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bptr,
            y.data_ptr<float>(),
            N, Cin, Lin,
            Cout, K,
            (int)stride, (int)padding, (int)dilation,
            Lout
        );
    }
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convtranspose1d_opt4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose1d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement for nn.ConvTranspose1d using a custom CUDA kernel (forward-only).
    Assumes input is CUDA, contiguous, and float32.
    Weight layout: (in_channels, out_channels, kernel_size).
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
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

        w = torch.empty(self.in_channels, self.out_channels, self.kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
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

        return self.custom_ops.conv_transpose1d_forward_cuda(
            x, w, b, self.stride, self.padding, self.dilation
        )