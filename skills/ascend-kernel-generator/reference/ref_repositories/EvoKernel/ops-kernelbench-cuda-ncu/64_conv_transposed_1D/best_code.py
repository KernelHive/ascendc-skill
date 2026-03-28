import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __forceinline__ __device__ float ld_g(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ float4 ld_g4(const float4* p) {
#if __CUDA_ARCH__ >= 350
    float4 v;
    v.x = __ldg(&p->x);
    v.y = __ldg(&p->y);
    v.z = __ldg(&p->z);
    v.w = __ldg(&p->w);
    return v;
#else
    return *p;
#endif
}

static __forceinline__ __device__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

static inline int div_up_i(int a, int b) { return (a + b - 1) / b; }
static inline int64_t div_up_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Repack PyTorch ConvTranspose1d weights from [Cin, Cout, K] (groups=1 case)
// into [Cin, K, Cout] for Cout-contiguous access.
__global__ void repack_w_cin_k_cout_g1(
    const float* __restrict__ w_in,  // [Cin, Cout, K]
    float* __restrict__ w_ok,        // [Cin, K, Cout]
    int Cin, int Cout, int K
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = Cin * Cout * K;
    if (idx >= total) return;

    int k = idx % K;
    int t = idx / K;
    int cout = t % Cout;
    int cin = t / Cout;

    // w_in index: ((cin*Cout + cout)*K + k)
    // w_ok index: ((cin*K + k)*Cout + cout)
    w_ok[((cin * K + k) * Cout + cout)] = w_in[((cin * Cout + cout) * K + k)];
}

// Fast path: groups=1, stride=1, dilation=1, output_padding=0, bias=None
// Uses repacked weights w_ok: [Cin, K, Cout]
// 2D block: threadIdx.x -> lout, threadIdx.y -> cout8 tile
template<int K_>
__global__ __launch_bounds__(256, 2)
void conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel(
    const float* __restrict__ x,     // [N, Cin, Lin]
    const float* __restrict__ w_ok,  // [Cin, K, Cout]
    float* __restrict__ y,           // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout,
    int padding,
    int Lout
) {
    int n = (int)blockIdx.y;
    int cout8 = ((int)blockIdx.z * (int)blockDim.y + (int)threadIdx.y) * 8;
    if (n >= N || cout8 >= Cout) return;

    const int x_n_base = n * Cin * Lin;

    // grid-stride loop over lout to improve occupancy for very large Lout
    for (int lout = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         lout < Lout;
         lout += (int)(gridDim.x * blockDim.x)) {

        float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f, acc4=0.f, acc5=0.f, acc6=0.f, acc7=0.f;

        const int A = lout + padding;

        // For stride=1,dilation=1:
        // lin = A - k, require 0<=lin<Lin => A-(Lin-1) <= k <= A
        int k_min = A - (Lin - 1);
        if (k_min < 0) k_min = 0;
        int k_max = A;
        if (k_max > (K_ - 1)) k_max = (K_ - 1);

        if (k_min <= k_max) {
            // start lin at k_min and decrement by 1 for each k
            int lin0 = A - k_min;

            for (int cin = 0; cin < Cin; ++cin) {
                const float* __restrict__ x_ptr = x + x_n_base + cin * Lin;
                const float* __restrict__ w_cin = w_ok + cin * (K_ * Cout);

                int lin = lin0;
                const float* __restrict__ w_k = w_cin + k_min * Cout + cout8;

                bool full8 = (cout8 + 7) < Cout;
                bool vec_ok = full8 && is_aligned_16(w_k) && is_aligned_16(w_ok);

                #pragma unroll
                for (int kk = 0; kk < K_; ++kk) {
                    int k = kk;
                    if (k < k_min || k > k_max) continue;

                    float xv = ld_g(x_ptr + (A - k));

                    const float* __restrict__ w_row = w_cin + k * Cout + cout8;
                    if (vec_ok) {
                        float4 v0 = ld_g4(reinterpret_cast<const float4*>(w_row));
                        float4 v1 = ld_g4(reinterpret_cast<const float4*>(w_row + 4));
                        acc0 = fmaf(xv, v0.x, acc0);
                        acc1 = fmaf(xv, v0.y, acc1);
                        acc2 = fmaf(xv, v0.z, acc2);
                        acc3 = fmaf(xv, v0.w, acc3);
                        acc4 = fmaf(xv, v1.x, acc4);
                        acc5 = fmaf(xv, v1.y, acc5);
                        acc6 = fmaf(xv, v1.z, acc6);
                        acc7 = fmaf(xv, v1.w, acc7);
                    } else {
                        if (cout8 + 0 < Cout) acc0 = fmaf(xv, ld_g(w_row + 0), acc0);
                        if (cout8 + 1 < Cout) acc1 = fmaf(xv, ld_g(w_row + 1), acc1);
                        if (cout8 + 2 < Cout) acc2 = fmaf(xv, ld_g(w_row + 2), acc2);
                        if (cout8 + 3 < Cout) acc3 = fmaf(xv, ld_g(w_row + 3), acc3);
                        if (cout8 + 4 < Cout) acc4 = fmaf(xv, ld_g(w_row + 4), acc4);
                        if (cout8 + 5 < Cout) acc5 = fmaf(xv, ld_g(w_row + 5), acc5);
                        if (cout8 + 6 < Cout) acc6 = fmaf(xv, ld_g(w_row + 6), acc6);
                        if (cout8 + 7 < Cout) acc7 = fmaf(xv, ld_g(w_row + 7), acc7);
                    }
                }
            }
        }

        int64_t base = ((int64_t)n * (int64_t)Cout + (int64_t)cout8) * (int64_t)Lout + (int64_t)lout;
        if (cout8 + 0 < Cout) y[base + 0LL * (int64_t)Lout] = acc0;
        if (cout8 + 1 < Cout) y[base + 1LL * (int64_t)Lout] = acc1;
        if (cout8 + 2 < Cout) y[base + 2LL * (int64_t)Lout] = acc2;
        if (cout8 + 3 < Cout) y[base + 3LL * (int64_t)Lout] = acc3;
        if (cout8 + 4 < Cout) y[base + 4LL * (int64_t)Lout] = acc4;
        if (cout8 + 5 < Cout) y[base + 5LL * (int64_t)Lout] = acc5;
        if (cout8 + 6 < Cout) y[base + 6LL * (int64_t)Lout] = acc6;
        if (cout8 + 7 < Cout) y[base + 7LL * (int64_t)Lout] = acc7;
    }
}

// General fallback kernel (supports groups/bias/stride/dilation/output_padding) using PyTorch weight layout [Cin, Cout/groups, K]
__global__ __launch_bounds__(256, 2)
void conv_transpose1d_forward_general_kernel(
    const float* __restrict__ x,      // [N, Cin, Lin]
    const float* __restrict__ w,      // [Cin, Cout_per_g, K]
    const float* __restrict__ b,      // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout, int K,
    int stride, int padding, int dilation, int output_padding,
    int groups,
    int Lout,
    int has_bias
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)N * (long long)Cout * (long long)Lout;
    if (idx >= total) return;

    int out_x = (int)(idx % Lout);
    long long tmp = idx / Lout;
    int oc = (int)(tmp % Cout);
    int n  = (int)(tmp / Cout);

    float acc = has_bias ? ld_g(b + oc) : 0.0f;

    int Cout_per_g = Cout / groups;
    int Cin_per_g  = Cin / groups;
    int g = oc / Cout_per_g;
    int ocg = oc - g * Cout_per_g;

    const long long x_n_base = (long long)n * (long long)Cin * (long long)Lin;

    for (int icg = 0; icg < Cin_per_g; ++icg) {
        int ic = g * Cin_per_g + icg;
        const float* x_ic = x + x_n_base + (long long)ic * (long long)Lin;
        const float* w_ic_ocg = w + ((long long)ic * (long long)Cout_per_g + (long long)ocg) * (long long)K;

        #pragma unroll 1
        for (int k = 0; k < K; ++k) {
            int t = out_x + padding - k * dilation;
            if (t < 0) continue;
            int r = t % stride;
            if (r != 0) continue;
            int in_x = t / stride;
            if ((unsigned)in_x >= (unsigned)Lin) continue;
            acc = fmaf(ld_g(x_ic + in_x), ld_g(w_ic_ocg + k), acc);
        }
    }

    y[((long long)n * (long long)Cout + (long long)oc) * (long long)Lout + (long long)out_x] = acc;
}

torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor x,           // [N, Cin, Lin]
    torch::Tensor w,           // [Cin, Cout/groups, K] (PyTorch layout)
    c10::optional<torch::Tensor> b_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t dilation,
    c10::optional<torch::Tensor> w_ok_opt // [Cin, K, Cout] for fast path (groups=1)
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 3, "x must be NCL (3D)");
    TORCH_CHECK(w.dim() == 3, "w must be (Cin, Cout/groups, K) (3D)");

    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(dilation > 0, "dilation must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");
    TORCH_CHECK(output_padding >= 0, "output_padding must be >= 0");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Lin = x.size(2);

    TORCH_CHECK(w.size(0) == Cin, "w.size(0) must equal Cin");
    const int64_t Cout_per_g = w.size(1);
    const int64_t K = w.size(2);
    TORCH_CHECK(K > 0, "kernel size must be > 0");

    TORCH_CHECK(Cin % groups == 0, "Cin must be divisible by groups");
    const int64_t Cout = Cout_per_g * groups;

    const int64_t Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    TORCH_CHECK(Lout > 0, "computed output length is non-positive");

    const bool has_bias = b_opt.has_value() && b_opt.value().defined();
    const float* b_ptr = nullptr;
    torch::Tensor b_c;
    if (has_bias) {
        auto b = b_opt.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "bias must be 1D of size Cout");
        b_c = b;
        b_ptr = b_c.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, Lout}, x.options());

    // Optimized fast path: groups=1, stride=1, dilation=1, output_padding=0, bias=None
    if (groups == 1 && stride == 1 && dilation == 1 && output_padding == 0 && !has_bias) {
        torch::Tensor w_ok;
        if (w_ok_opt.has_value() && w_ok_opt.value().defined()) {
            w_ok = w_ok_opt.value();
            CHECK_INPUT(w_ok);
            TORCH_CHECK(w_ok.dim() == 3, "w_ok must be [Cin, K, Cout]");
            TORCH_CHECK(w_ok.size(0) == Cin && w_ok.size(1) == K && w_ok.size(2) == Cout, "w_ok shape mismatch");
        } else {
            // Build w_ok on the fly if not provided (slower; Python should cache it).
            w_ok = torch::empty({Cin, K, Cout}, w.options());
            const int threads = 256;
            const int blocks = div_up_i((int)(Cin * Cout * K), threads);
            repack_w_cin_k_cout_g1<<<blocks, threads, 0, stream>>>(
                w.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                (int)Cin, (int)Cout, (int)K
            );
        }

        // threads.x over lout, threads.y over cout8 tiles; 128*2=256 threads
        dim3 threads(128, 2, 1);
        int gx = (int)div_up_i64(Lout, (int64_t)threads.x);
        if (gx > 4096) gx = 4096; // keep grid reasonable
        dim3 blocks((unsigned)gx, (unsigned)N, (unsigned)div_up_i(div_up_i((int)Cout, 8), (int)threads.y));

        // Specialize small K to reduce loop/branch overhead. Prompt uses K=3.
        if (K == 3) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<3><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else if (K == 1) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<1><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else if (K == 2) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<2><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else if (K == 4) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<4><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else if (K == 5) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<5><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else if (K == 6) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<6><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else if (K == 7) {
            conv_t1d_g1_s1_d1_op0_nobias_wok_cout8_k_kernel<7><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout,
                (int)padding,
                (int)Lout
            );
        } else {
            // For larger K, fall back to general kernel (still correct, but slower).
            const int threads_g = 256;
            const long long total = (long long)N * (long long)Cout * (long long)Lout;
            const int blocks_g = (int)((total + threads_g - 1) / threads_g);
            conv_transpose1d_forward_general_kernel<<<blocks_g, threads_g, 0, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                nullptr,
                y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Lin,
                (int)Cout, (int)K,
                (int)stride, (int)padding, (int)dilation, (int)output_padding,
                (int)groups,
                (int)Lout,
                0
            );
        }
        return y;
    }

    // General fallback
    const int threads = 256;
    const long long total = (long long)N * (long long)Cout * (long long)Lout;
    const int blocks = (int)((total + threads - 1) / threads);

    conv_transpose1d_forward_general_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b_ptr,
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Lin,
        (int)Cout, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)output_padding,
        (int)groups,
        (int)Lout,
        has_bias ? 1 : 0
    );

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
    int64_t output_padding,
    int64_t groups,
    int64_t dilation,
    c10::optional<torch::Tensor> w_ok_opt
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convtranspose1d_opt_inc4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose1d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Forward-only ConvTranspose1d replacement via custom CUDA kernel.

    Fast path (optimized):
      groups=1, stride=1, dilation=1, output_padding=0, bias=None
      Uses cached repacked weights [Cin, K, Cout] to enable coalesced Cout loads and Cout8 vectorization.

    General path:
      supports groups/bias/stride/dilation/output_padding via a generic CUDA kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.groups = int(groups)
        self.dilation = int(dilation)

        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        w = torch.empty(
            self.in_channels,
            self.out_channels // self.groups,
            self.kernel_size,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32)) if bias else None
        self.custom_ops = custom_ops_lib

        # Cached repacked weights for fast path (groups=1 only): [Cin, K, Cout]
        self.register_buffer("_w_ok", None, persistent=False)
        self.register_buffer("_w_ok_meta", torch.zeros(4, dtype=torch.int64), persistent=False)  # [ptr, Cin, K, Cout]

    @torch.no_grad()
    def _ensure_w_ok(self, device):
        if not (self.groups == 1 and self.stride == 1 and self.dilation == 1 and self.output_padding == 0 and self.bias is None):
            self._w_ok = None
            return

        w = self.weight
        if not w.is_cuda or w.device != device:
            w = w.to(device=device)

        w = w.contiguous()
        Cin = int(w.shape[0])
        Cout = int(w.shape[1])
        K = int(w.shape[2])

        # Invalidate if shape changed or weight storage pointer changed (covers most inference/training updates)
        ptr = int(w.untyped_storage().data_ptr())
        meta = self._w_ok_meta
        need = (
            (self._w_ok is None)
            or (not self._w_ok.is_cuda)
            or (self._w_ok.device != device)
            or (int(meta[0].item()) != ptr)
            or (int(meta[1].item()) != Cin)
            or (int(meta[2].item()) != K)
            or (int(meta[3].item()) != Cout)
            or (list(self._w_ok.shape) != [Cin, K, Cout])
        )

        if need:
            # Repack via permute+contiguous (uses PyTorch optimized transpose kernels)
            self._w_ok = w.detach().permute(0, 2, 1).contiguous()
            meta[0] = ptr
            meta[1] = Cin
            meta[2] = K
            meta[3] = Cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        if not w.is_cuda or w.device != x.device:
            w = w.to(device=x.device)
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        b = self.bias
        if b is not None:
            if not b.is_cuda or b.device != x.device:
                b = b.to(device=x.device)
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()

        self._ensure_w_ok(x.device)
        w_ok = self._w_ok

        return self.custom_ops.conv_transpose1d_forward_cuda(
            x, w, b,
            self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            w_ok
        )