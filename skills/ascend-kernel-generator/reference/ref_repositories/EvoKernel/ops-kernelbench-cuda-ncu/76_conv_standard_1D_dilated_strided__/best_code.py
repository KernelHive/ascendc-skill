import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

static inline int64_t compute_lout(int64_t Lin, int64_t K, int64_t stride, int64_t dilation) {
    int64_t effective = dilation * (K - 1) + 1;
    if (Lin < effective) return 0;
    return (Lin - effective) / stride + 1;
}

__device__ __forceinline__ float ro_load_f32(const float* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

template <bool HAS_BIAS>
__global__ __launch_bounds__(128, 4) void conv1d_fwd_generic_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int Lin,
    int Cout, int K,
    int stride, int dilation,
    int Lout
) {
    int tid = (int)threadIdx.x;
    int l0  = (int)blockIdx.x * (int)blockDim.x + tid;

    int nc = (int)blockIdx.y;
    int n  = nc / Cout;
    int co = nc - n * Cout;
    if (n >= N || co >= Cout) return;

    int l = l0;
    int grid_stride = (int)blockDim.x * (int)gridDim.x;

    const float* __restrict__ w_base = w + (co * Cin * K);

    for (; l < Lout; l += grid_stride) {
        int base_in = l * stride;
        float acc = 0.0f;

        for (int ci = 0; ci < Cin; ++ci) {
            const float* __restrict__ x_ptr = x + ((n * Cin + ci) * Lin);
            const float* __restrict__ w_ptr = w_base + (ci * K);

#pragma unroll 1
            for (int k = 0; k < K; ++k) {
                int li = base_in + k * dilation;
                float xv = ((unsigned)li < (unsigned)Lin) ? ro_load_f32(x_ptr + li) : 0.0f;
                acc = fmaf(xv, ro_load_f32(w_ptr + k), acc);
            }
        }

        if constexpr (HAS_BIAS) acc += ro_load_f32(b + co);
        y[((n * Cout + co) * Lout) + l] = acc;
    }
}

// Specialized K=3 kernel: block computes a tile of output channels and a span of Lout.
// - 256 threads/block (8 warps) for better latency hiding.
// - co tile = 4 channels per block (fits common Cout=128 nicely).
// - Each thread computes 4 output positions (ILP).
// - Stage weights for each co into shared memory: [CO_TILE][Cin][3].
// - Split into interior (no bounds checks) and tail (checks) regions.
template <bool HAS_BIAS, int CO_TILE, int ILP>
__global__ __launch_bounds__(256, 2) void conv1d_fwd_k3_cotile_ilp_f32_kernel(
    const float* __restrict__ x,   // [N, Cin, Lin]
    const float* __restrict__ w,   // [Cout, Cin, 3]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout,
    int stride, int dilation,
    int Lout
) {
    extern __shared__ float shw[]; // size = CO_TILE * Cin * 3
    int tid = (int)threadIdx.x;

    int n = (int)blockIdx.y;
    int co0 = (int)blockIdx.z * CO_TILE;
    if (n >= N) return;

    // Stage weights into shared memory
    // Layout: shw[(t * Cin + ci) * 3 + k]
    int w_elems = CO_TILE * Cin * 3;
    for (int idx = tid; idx < w_elems; idx += (int)blockDim.x) {
        int k = idx % 3;
        int tmp = idx / 3;
        int ci = tmp % Cin;
        int t  = tmp / Cin;
        int co = co0 + t;
        float val = 0.0f;
        if (co < Cout) {
            const float* wp = w + ((co * Cin + ci) * 3 + k);
            val = ro_load_f32(wp);
        }
        shw[idx] = val;
    }
    __syncthreads();

    // Base Lout index for this block tile
    int base_l = (int)blockIdx.x * ((int)blockDim.x * ILP) + tid;

    // Prepare l positions and bases
    int l[ILP];
    int base_in[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
        l[i] = base_l + i * (int)blockDim.x;
        base_in[i] = l[i] * stride;
    }

    // Interior condition: base_in + 2*dilation < Lin  => base_in <= Lin - 1 - 2*dilation
    int interior_max_base = Lin - 1 - 2 * dilation;

    // Accumulators per co tile and ILP
    float acc[CO_TILE][ILP];
#pragma unroll
    for (int t = 0; t < CO_TILE; ++t) {
#pragma unroll
        for (int i = 0; i < ILP; ++i) acc[t][i] = 0.0f;
    }

    const float* __restrict__ x_n_base = x + (n * Cin * Lin);

    // Main accumulation over Cin
    for (int ci = 0; ci < Cin; ++ci) {
        const float* __restrict__ x_ptr = x_n_base + ci * Lin;

        // Load weights for all co in tile from shared
        float w0[CO_TILE], w1[CO_TILE], w2[CO_TILE];
#pragma unroll
        for (int t = 0; t < CO_TILE; ++t) {
            int off = (t * Cin + ci) * 3;
            w0[t] = shw[off + 0];
            w1[t] = shw[off + 1];
            w2[t] = shw[off + 2];
        }

        // For each ILP output, do interior fast path when in range
#pragma unroll
        for (int i = 0; i < ILP; ++i) {
            int li0 = base_in[i];
            if ((unsigned)l[i] >= (unsigned)Lout) continue;

            float x0, x1, x2;
            if (li0 <= interior_max_base && li0 >= 0) {
                // All in bounds
                x0 = ro_load_f32(x_ptr + li0);
                x1 = ro_load_f32(x_ptr + (li0 + dilation));
                x2 = ro_load_f32(x_ptr + (li0 + 2 * dilation));
            } else {
                int li1 = li0 + dilation;
                int li2 = li0 + 2 * dilation;
                x0 = ((unsigned)li0 < (unsigned)Lin) ? ro_load_f32(x_ptr + li0) : 0.0f;
                x1 = ((unsigned)li1 < (unsigned)Lin) ? ro_load_f32(x_ptr + li1) : 0.0f;
                x2 = ((unsigned)li2 < (unsigned)Lin) ? ro_load_f32(x_ptr + li2) : 0.0f;
            }

#pragma unroll
            for (int t = 0; t < CO_TILE; ++t) {
                acc[t][i] = fmaf(x0, w0[t], acc[t][i]);
                acc[t][i] = fmaf(x1, w1[t], acc[t][i]);
                acc[t][i] = fmaf(x2, w2[t], acc[t][i]);
            }
        }
    }

    // Write out (optionally add bias)
#pragma unroll
    for (int t = 0; t < CO_TILE; ++t) {
        int co = co0 + t;
        if (co >= Cout) continue;

        float biasv = 0.0f;
        if constexpr (HAS_BIAS) biasv = ro_load_f32(b + co);

        float* __restrict__ y_ptr = y + ((n * Cout + co) * Lout);

#pragma unroll
        for (int i = 0; i < ILP; ++i) {
            if ((unsigned)l[i] < (unsigned)Lout) {
                float outv = acc[t][i] + (HAS_BIAS ? biasv : 0.0f);
                y_ptr[l[i]] = outv;
            }
        }
    }
}

torch::Tensor conv1d_forward_cuda(torch::Tensor x,
                                 torch::Tensor w,
                                 c10::optional<torch::Tensor> b_opt,
                                 int64_t stride,
                                 int64_t dilation) {
    TORCH_CHECK(x.is_cuda(), "conv1d_forward_cuda: x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "conv1d_forward_cuda: w must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "conv1d_forward_cuda: x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "conv1d_forward_cuda: w must be float32");
    TORCH_CHECK(x.is_contiguous(), "conv1d_forward_cuda: x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "conv1d_forward_cuda: w must be contiguous");
    TORCH_CHECK(x.dim() == 3, "conv1d_forward_cuda: x must be [N, Cin, Lin]");
    TORCH_CHECK(w.dim() == 3, "conv1d_forward_cuda: w must be [Cout, Cin, K]");
    TORCH_CHECK(stride >= 1, "conv1d_forward_cuda: stride must be >= 1");
    TORCH_CHECK(dilation >= 1, "conv1d_forward_cuda: dilation must be >= 1");

    int64_t N64 = x.size(0);
    int64_t Cin64 = x.size(1);
    int64_t Lin64 = x.size(2);
    int64_t Cout64 = w.size(0);
    int64_t Cin_w64 = w.size(1);
    int64_t K64 = w.size(2);
    TORCH_CHECK(Cin64 == Cin_w64, "conv1d_forward_cuda: Cin mismatch between x and w");

    const float* b_ptr = nullptr;
    torch::Tensor b;
    bool has_bias = false;
    if (b_opt.has_value() && b_opt.value().defined()) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "conv1d_forward_cuda: bias must be CUDA");
        TORCH_CHECK(b.scalar_type() == at::kFloat, "conv1d_forward_cuda: bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "conv1d_forward_cuda: bias must be contiguous");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout64, "conv1d_forward_cuda: bias must be [Cout]");
        b_ptr = b.data_ptr<float>();
        has_bias = true;
    }

    int64_t Lout64 = compute_lout(Lin64, K64, stride, dilation);
    auto y = torch::empty({N64, Cout64, Lout64}, x.options());
    if (Lout64 == 0) return y;

    int N = (int)N64;
    int Cin = (int)Cin64;
    int Lin = (int)Lin64;
    int Cout = (int)Cout64;
    int K = (int)K64;
    int Lout = (int)Lout64;

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (K == 3) {
        constexpr int THREADS = 256;
        constexpr int CO_TILE = 4;
        constexpr int ILP = 4;

        dim3 block(THREADS, 1, 1);
        int grid_x = (Lout + (THREADS * ILP) - 1) / (THREADS * ILP);
        int grid_z = (Cout + CO_TILE - 1) / CO_TILE;
        dim3 grid((unsigned)grid_x, (unsigned)N, (unsigned)grid_z);

        size_t shmem = (size_t)CO_TILE * (size_t)Cin * 3u * sizeof(float);

        if (has_bias) {
            conv1d_fwd_k3_cotile_ilp_f32_kernel<true, CO_TILE, ILP><<<grid, block, shmem, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, Cin, Lin,
                Cout,
                (int)stride, (int)dilation,
                Lout
            );
        } else {
            conv1d_fwd_k3_cotile_ilp_f32_kernel<false, CO_TILE, ILP><<<grid, block, shmem, stream>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                nullptr,
                y.data_ptr<float>(),
                N, Cin, Lin,
                Cout,
                (int)stride, (int)dilation,
                Lout
            );
        }
        return y;
    }

    const int threads = 128;
    int grid_x = (int)((Lout + threads - 1) / threads);
    if (grid_x > 32768) grid_x = 32768;
    dim3 block(threads);
    dim3 grid((unsigned)grid_x, (unsigned)(N * Cout));

    if (has_bias) {
        conv1d_fwd_generic_f32_kernel<true><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b_ptr,
            y.data_ptr<float>(),
            N, Cin, Lin,
            Cout, K,
            (int)stride, (int)dilation,
            Lout
        );
    } else {
        conv1d_fwd_generic_f32_kernel<false><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            y.data_ptr<float>(),
            N, Cin, Lin,
            Cout, K,
            (int)stride, (int)dilation,
            Lout
        );
    }

    return y;
}
"""

conv1d_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv1d_forward_cuda(torch::Tensor x,
                                 torch::Tensor w,
                                 c10::optional<torch::Tensor> b_opt,
                                 int64_t stride,
                                 int64_t dilation);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv1d_dilated_strided_opt6_cotile_shw_ilp",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_cuda_source,
    functions=["conv1d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
)


class ModelNew(nn.Module):
    """
    Replacement for nn.Conv1d using a custom CUDA kernel (forward-only).
    Assumes padding=0, groups=1. Supports float32 CUDA contiguous input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.use_bias = bool(bias)

        w = torch.empty(self.out_channels, self.in_channels, self.kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if self.use_bias:
            b = torch.empty(self.out_channels, dtype=torch.float32)
            fan_in = self.in_channels * self.kernel_size
            bound = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter("bias", None)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input.")
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

        if self.bias is None:
            return self.custom_ops_lib.conv1d_forward_cuda(x, w, None, self.stride, self.dilation)

        b = self.bias
        if not b.is_cuda:
            b = b.to(device=x.device)
        if b.dtype != torch.float32:
            b = b.float()
        if not b.is_contiguous():
            b = b.contiguous()

        return self.custom_ops_lib.conv1d_forward_cuda(x, w, b, self.stride, self.dilation)