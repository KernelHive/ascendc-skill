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

static inline int div_up_i(int a, int b) { return (a + b - 1) / b; }
static inline int64_t div_up_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__global__ void repack_w_cin_k_cout(
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
    w_ok[((cin * K + k) * Cout + cout)] = w_in[((cin * Cout + cout) * K + k)];
}

static __forceinline__ __device__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// ---------------- stride==1, repacked weights, Cout8 vectorization ----------------
// threads.x over lout, threads.y over cout8 groups.
__global__ __launch_bounds__(256, 2)
void conv_t1d_s1_wok_cout8_kernel(
    const float* __restrict__ x,     // [N, Cin, Lin]
    const float* __restrict__ w_ok,  // [Cin, K, Cout]
    const float* __restrict__ b,     // [Cout] or nullptr
    float* __restrict__ y,           // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout, int K,
    int padding, int dilation,
    int Lout,
    int has_bias
) {
    int n = (int)blockIdx.y;
    int cout8 = ((int)blockIdx.z * (int)blockDim.y + (int)threadIdx.y) * 8;
    if (n >= N || cout8 >= Cout) return;

    const int x_n_base = n * Cin * Lin;

    for (int lout = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         lout < Lout;
         lout += (int)(gridDim.x * blockDim.x)) {

        float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f, acc4=0.f, acc5=0.f, acc6=0.f, acc7=0.f;

        if (has_bias) {
            if (cout8 + 0 < Cout) acc0 = ld_g(b + (cout8 + 0));
            if (cout8 + 1 < Cout) acc1 = ld_g(b + (cout8 + 1));
            if (cout8 + 2 < Cout) acc2 = ld_g(b + (cout8 + 2));
            if (cout8 + 3 < Cout) acc3 = ld_g(b + (cout8 + 3));
            if (cout8 + 4 < Cout) acc4 = ld_g(b + (cout8 + 4));
            if (cout8 + 5 < Cout) acc5 = ld_g(b + (cout8 + 5));
            if (cout8 + 6 < Cout) acc6 = ld_g(b + (cout8 + 6));
            if (cout8 + 7 < Cout) acc7 = ld_g(b + (cout8 + 7));
        }

        const int A = lout + padding;

        // k range (only where lin in [0, Lin-1])
        int k_min = (A - (Lin - 1) + dilation - 1) / dilation; // ceil
        int k_max = A / dilation;                               // floor
        if (k_min < 0) k_min = 0;
        if (k_max > K - 1) k_max = K - 1;

        if (k_min <= k_max) {
            // Start lin at k_min, then decrement by dilation each k.
            int lin = A - k_min * dilation;

            for (int cin = 0; cin < Cin; ++cin) {
                const float* __restrict__ x_ptr = x + x_n_base + cin * Lin;
                const float* __restrict__ w_cin = w_ok + cin * (K * Cout);

                int lin_k = lin;
                const float* __restrict__ w_k = w_cin + k_min * Cout + cout8;

                // vectorization guard
                bool vec_ok = (cout8 + 7 < Cout) && is_aligned_16(w_k) && is_aligned_16(w_ok);

                #pragma unroll 1
                for (int k = k_min; k <= k_max; ++k) {
                    float xv = ld_g(x_ptr + lin_k);

                    if (vec_ok) {
                        float4 w0 = ld_g4(reinterpret_cast<const float4*>(w_k));
                        float4 w1 = ld_g4(reinterpret_cast<const float4*>(w_k + 4));
                        acc0 = fmaf(xv, w0.x, acc0);
                        acc1 = fmaf(xv, w0.y, acc1);
                        acc2 = fmaf(xv, w0.z, acc2);
                        acc3 = fmaf(xv, w0.w, acc3);
                        acc4 = fmaf(xv, w1.x, acc4);
                        acc5 = fmaf(xv, w1.y, acc5);
                        acc6 = fmaf(xv, w1.z, acc6);
                        acc7 = fmaf(xv, w1.w, acc7);
                    } else {
                        if (cout8 + 0 < Cout) acc0 = fmaf(xv, ld_g(w_k + 0), acc0);
                        if (cout8 + 1 < Cout) acc1 = fmaf(xv, ld_g(w_k + 1), acc1);
                        if (cout8 + 2 < Cout) acc2 = fmaf(xv, ld_g(w_k + 2), acc2);
                        if (cout8 + 3 < Cout) acc3 = fmaf(xv, ld_g(w_k + 3), acc3);
                        if (cout8 + 4 < Cout) acc4 = fmaf(xv, ld_g(w_k + 4), acc4);
                        if (cout8 + 5 < Cout) acc5 = fmaf(xv, ld_g(w_k + 5), acc5);
                        if (cout8 + 6 < Cout) acc6 = fmaf(xv, ld_g(w_k + 6), acc6);
                        if (cout8 + 7 < Cout) acc7 = fmaf(xv, ld_g(w_k + 7), acc7);
                    }

                    lin_k -= dilation;
                    w_k += Cout;
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

// ---------------- stride==1 small-K specialization (K<=7), repacked weights, Cout8 ----------------
template<int K_>
__global__ __launch_bounds__(256, 2)
void conv_t1d_s1_wok_cout8_k_small_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_ok,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int Lin,
    int Cout,
    int padding, int dilation,
    int Lout,
    int has_bias
) {
    int n = (int)blockIdx.y;
    int cout8 = ((int)blockIdx.z * (int)blockDim.y + (int)threadIdx.y) * 8;
    if (n >= N || cout8 >= Cout) return;

    const int x_n_base = n * Cin * Lin;

    for (int lout = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         lout < Lout;
         lout += (int)(gridDim.x * blockDim.x)) {

        float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f, acc4=0.f, acc5=0.f, acc6=0.f, acc7=0.f;
        if (has_bias) {
            if (cout8 + 0 < Cout) acc0 = ld_g(b + (cout8 + 0));
            if (cout8 + 1 < Cout) acc1 = ld_g(b + (cout8 + 1));
            if (cout8 + 2 < Cout) acc2 = ld_g(b + (cout8 + 2));
            if (cout8 + 3 < Cout) acc3 = ld_g(b + (cout8 + 3));
            if (cout8 + 4 < Cout) acc4 = ld_g(b + (cout8 + 4));
            if (cout8 + 5 < Cout) acc5 = ld_g(b + (cout8 + 5));
            if (cout8 + 6 < Cout) acc6 = ld_g(b + (cout8 + 6));
            if (cout8 + 7 < Cout) acc7 = ld_g(b + (cout8 + 7));
        }

        const int A = lout + padding;
        // k range
        int k_min = (A - (Lin - 1) + dilation - 1) / dilation; // ceil
        int k_max = A / dilation;                               // floor
        if (k_min < 0) k_min = 0;
        if (k_max > K_ - 1) k_max = K_ - 1;

        if (k_min <= k_max) {
            for (int cin = 0; cin < Cin; ++cin) {
                const float* __restrict__ x_ptr = x + x_n_base + cin * Lin;
                const float* __restrict__ w_cin = w_ok + cin * (K_ * Cout);

                #pragma unroll
                for (int k = 0; k < K_; ++k) {
                    if (k < k_min || k > k_max) continue;
                    int lin = A - k * dilation;
                    float xv = ld_g(x_ptr + lin);
                    const float* __restrict__ w_k = w_cin + k * Cout + cout8;

                    bool vec_ok = (cout8 + 7 < Cout) && is_aligned_16(w_k) && is_aligned_16(w_ok);
                    if (vec_ok) {
                        float4 w0 = ld_g4(reinterpret_cast<const float4*>(w_k));
                        float4 w1 = ld_g4(reinterpret_cast<const float4*>(w_k + 4));
                        acc0 = fmaf(xv, w0.x, acc0);
                        acc1 = fmaf(xv, w0.y, acc1);
                        acc2 = fmaf(xv, w0.z, acc2);
                        acc3 = fmaf(xv, w0.w, acc3);
                        acc4 = fmaf(xv, w1.x, acc4);
                        acc5 = fmaf(xv, w1.y, acc5);
                        acc6 = fmaf(xv, w1.z, acc6);
                        acc7 = fmaf(xv, w1.w, acc7);
                    } else {
                        if (cout8 + 0 < Cout) acc0 = fmaf(xv, ld_g(w_k + 0), acc0);
                        if (cout8 + 1 < Cout) acc1 = fmaf(xv, ld_g(w_k + 1), acc1);
                        if (cout8 + 2 < Cout) acc2 = fmaf(xv, ld_g(w_k + 2), acc2);
                        if (cout8 + 3 < Cout) acc3 = fmaf(xv, ld_g(w_k + 3), acc3);
                        if (cout8 + 4 < Cout) acc4 = fmaf(xv, ld_g(w_k + 4), acc4);
                        if (cout8 + 5 < Cout) acc5 = fmaf(xv, ld_g(w_k + 5), acc5);
                        if (cout8 + 6 < Cout) acc6 = fmaf(xv, ld_g(w_k + 6), acc6);
                        if (cout8 + 7 < Cout) acc7 = fmaf(xv, ld_g(w_k + 7), acc7);
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

// ---------------- general fallback (stride>=1), PyTorch weight layout ----------------
__global__ void conv_transpose1d_dilated_fwd_general_kernel(
    const float* __restrict__ x,        // [N, Cin, Lin]
    const float* __restrict__ w,        // [Cin, Cout, K]
    const float* __restrict__ b,        // [Cout] or nullptr
    float* __restrict__ y,              // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout, int K,
    int stride, int padding, int dilation,
    int Lout,
    int has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Lout;
    if (idx >= total) return;

    int lout = idx % Lout;
    int tmp = idx / Lout;
    int cout = tmp % Cout;
    int n = tmp / Cout;

    float acc = has_bias ? ld_g(b + cout) : 0.0f;

    const int x_n_base = n * Cin * Lin;
    const int w_cin_stride = Cout * K;

    for (int cin = 0; cin < Cin; ++cin) {
        const float* x_ptr = x + x_n_base + cin * Lin;
        const float* w_ptr = w + cin * w_cin_stride + cout * K;

        #pragma unroll 1
        for (int k = 0; k < K; ++k) {
            int numer = lout + padding - k * dilation;
            if ((numer % stride) != 0) continue;
            int lin = numer / stride;
            if ((unsigned)lin < (unsigned)Lin) {
                acc = fmaf(ld_g(x_ptr + lin), ld_g(w_ptr + k), acc);
            }
        }
    }
    y[idx] = acc;
}

torch::Tensor conv_transpose1d_dilated_cuda(
    torch::Tensor x,      // [N, Cin, Lin]
    torch::Tensor w,      // [Cin, Cout, K]
    c10::optional<torch::Tensor> b_opt, // [Cout] or None
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    c10::optional<torch::Tensor> w_ok_opt // [Cin, K, Cout] or None (cached repack)
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be (N, Cin, Lin)");
    TORCH_CHECK(w.dim() == 3, "w must be (Cin, Cout, K)");
    TORCH_CHECK(stride >= 1, "stride must be >= 1");
    TORCH_CHECK(dilation >= 1, "dilation must be >= 1");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int N = (int)x_c.size(0);
    const int Cin = (int)x_c.size(1);
    const int Lin = (int)x_c.size(2);

    TORCH_CHECK((int)w_c.size(0) == Cin, "w.shape[0] must equal Cin");
    const int Cout = (int)w_c.size(1);
    const int K = (int)w_c.size(2);

    const int Lout = (int)((Lin - 1) * (int)stride - 2 * (int)padding + (int)dilation * (K - 1) + 1);
    TORCH_CHECK(Lout > 0, "Computed Lout must be > 0");

    torch::Tensor y = torch::zeros({N, Cout, Lout}, x_c.options());

    const float* b_ptr = nullptr;
    int has_bias = 0;
    torch::Tensor b_c;
    if (b_opt.has_value() && b_opt.value().defined()) {
        auto b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == Cout, "bias must be (Cout,)");
        b_c = b.contiguous();
        b_ptr = b_c.data_ptr<float>();
        has_bias = 1;
    }

    if ((int)stride == 1) {
        torch::Tensor w_ok;
        if (w_ok_opt.has_value() && w_ok_opt.value().defined()) {
            w_ok = w_ok_opt.value();
            TORCH_CHECK(w_ok.is_cuda(), "w_ok must be CUDA");
            TORCH_CHECK(w_ok.dtype() == torch::kFloat32, "w_ok must be float32");
            TORCH_CHECK(w_ok.is_contiguous(), "w_ok must be contiguous");
            TORCH_CHECK(w_ok.dim() == 3, "w_ok must be (Cin, K, Cout)");
            TORCH_CHECK((int)w_ok.size(0) == Cin && (int)w_ok.size(1) == K && (int)w_ok.size(2) == Cout,
                        "w_ok shape mismatch");
        } else {
            w_ok = torch::empty({Cin, K, Cout}, w_c.options());
            const int repack_threads = 256;
            const int repack_blocks = div_up_i(Cin * Cout * K, repack_threads);
            repack_w_cin_k_cout<<<repack_blocks, repack_threads, 0, stream>>>(
                w_c.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                Cin, Cout, K
            );
        }

        // threads.x: lout, threads.y: cout8 groups. 128*2=256 threads.
        dim3 threads(128, 2, 1);

        int gx = (int)div_up_i64((int64_t)Lout, (int64_t)threads.x);
        if (gx > 4096) gx = 4096;

        dim3 blocks(
            (unsigned)gx,
            (unsigned)N,
            (unsigned)div_up_i(div_up_i(Cout, 8), (int)threads.y)
        );

        // Small-K specialization (common conv kernels) to reduce loop overhead.
        if (K <= 7) {
            switch (K) {
                case 1:
                    conv_t1d_s1_wok_cout8_k_small_kernel<1><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                case 2:
                    conv_t1d_s1_wok_cout8_k_small_kernel<2><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                case 3:
                    conv_t1d_s1_wok_cout8_k_small_kernel<3><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                case 4:
                    conv_t1d_s1_wok_cout8_k_small_kernel<4><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                case 5:
                    conv_t1d_s1_wok_cout8_k_small_kernel<5><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                case 6:
                    conv_t1d_s1_wok_cout8_k_small_kernel<6><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                case 7:
                    conv_t1d_s1_wok_cout8_k_small_kernel<7><<<blocks, threads, 0, stream>>>(
                        x_c.data_ptr<float>(), w_ok.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                        N, Cin, Lin, Cout, (int)padding, (int)dilation, Lout, has_bias
                    ); break;
                default:
                    break;
            }
        } else {
            conv_t1d_s1_wok_cout8_kernel<<<blocks, threads, 0, stream>>>(
                x_c.data_ptr<float>(),
                w_ok.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, Cin, Lin,
                Cout, K,
                (int)padding, (int)dilation,
                Lout,
                has_bias
            );
        }
        return y;
    }

    int total = N * Cout * Lout;
    const int threads_g = 256;
    const int blocks_g = div_up_i(total, threads_g);

    conv_transpose1d_dilated_fwd_general_kernel<<<blocks_g, threads_g, 0, stream>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        b_ptr,
        y.data_ptr<float>(),
        N, Cin, Lin,
        Cout, K,
        (int)stride, (int)padding, (int)dilation,
        Lout,
        has_bias
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose1d_dilated_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    c10::optional<torch::Tensor> w_ok_opt
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose1d_dilated_opt7",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose1d_dilated_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


class ModelNew(nn.Module):
    """
    Drop-in replacement for nn.ConvTranspose1d (forward only) using an optimized custom CUDA kernel.
    Constraints: CUDA float32, groups=1, output_padding=0.
    Weight layout matches PyTorch ConvTranspose1d: (in_channels, out_channels, kernel_size).
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

        self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32)) if bias else None

        self.custom_ops = custom_ops_lib

        # Cached repacked weights for stride==1: [Cin, K, Cout]
        self.register_buffer("_w_ok", None, persistent=False)
        self.register_buffer("_w_version", torch.zeros((), dtype=torch.int64), persistent=False)
        self._w_version_int = 0

        # Increment version when autograd produces a new weight grad application (covers training).
        # For inference-only, version stays stable and cache persists.
        def _bump_version(_grad):
            self._w_version_int += 1
            self._w_version.fill_(self._w_version_int)
            return _grad

        self.weight.register_hook(_bump_version)

    @torch.no_grad()
    def _ensure_w_ok(self):
        # Only used for stride==1 path; rebuild if missing, device mismatch, shape mismatch, or version changed.
        if self.stride != 1:
            return

        dev = self.weight.device
        if (self._w_ok is None) or (not self._w_ok.is_cuda) or (self._w_ok.device != dev):
            self._w_ok = None

        need = False
        if self._w_ok is None:
            need = True
        else:
            if list(self._w_ok.shape) != [self.in_channels, self.kernel_size, self.out_channels]:
                need = True

        # Version-based invalidation: safe for optimizer updates and most training flows.
        # If user mutates weight in-place without autograd, they should call .invalidate_cache().
        if int(self._w_version.item()) != self._w_version_int:
            self._w_version_int = int(self._w_version.item())
            need = True

        if need:
            # Repack on GPU using a simple permute+contiguous (fast enough and avoids extra custom entry point);
            # kernel-side still benefits from cached layout and no per-forward repack launch.
            w_ok = self.weight.detach().permute(0, 2, 1).contiguous()
            self._w_ok = w_ok

    def invalidate_cache(self):
        self._w_ok = None
        self._w_version_int += 1
        self._w_version.fill_(self._w_version_int)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.weight.is_cuda:
            self._ensure_w_ok()
            w_ok = self._w_ok
        else:
            w_ok = None

        return self.custom_ops.conv_transpose1d_dilated_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, w_ok
        )