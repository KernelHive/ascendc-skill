import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

avg_pool1d_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_DTYPE
#define CHECK_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

static inline int64_t div_up_int64_host(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// -------------------- General baseline (any stride) --------------------
__global__ __launch_bounds__(256, 2)
void avg_pool1d_forward_baseline_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int Lin,
    int Lout,
    int kL,
    int sL,
    int pL
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)Lout;
    if (idx >= total) return;

    int lo = (int)(idx % Lout);
    int tmp = (int)(idx / Lout);
    int c = (int)(tmp % C);
    int n = (int)(tmp / C);

    int lstart = lo * sL - pL;
    int lend = lstart + kL;

    int l0 = lstart < 0 ? 0 : lstart;
    int l1 = lend > Lin ? Lin : lend;

    const int64_t base = ((int64_t)n * (int64_t)C + c) * (int64_t)Lin;
    float sum = 0.0f;

    int li = l0;
    #pragma unroll 4
    for (; li + 3 < l1; li += 4) {
        sum += ldg_f32(x + base + (int64_t)(li + 0));
        sum += ldg_f32(x + base + (int64_t)(li + 1));
        sum += ldg_f32(x + base + (int64_t)(li + 2));
        sum += ldg_f32(x + base + (int64_t)(li + 3));
    }
    for (; li < l1; ++li) sum += ldg_f32(x + base + (int64_t)li);

    y[idx] = sum / (float)kL; // count_include_pad=True
}

// -------------------- Specialized fast path: k=8, s=1, p=4 (ILP: 2 outputs/thread) --------------------
// Mapping: threads iterate over lo (length) first, then (n,c) packed into nc.
// Each thread computes lo and lo+1 for same (n,c) when in range.
// Keeps vectorized float4 loads for interior windows when aligned.
__global__ __launch_bounds__(256, 2)
void avg_pool1d_forward_k8s1p4_kernel_ilp2(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int Lin, int Lout
) {
    // Lout = Lin + 1 for k=8,s=1,p=4
    const int64_t NC = (int64_t)N * (int64_t)C;
    const int64_t lo_pairs = (int64_t)(Lout + 1) / 2; // ceil(Lout/2)
    const int64_t total_pairs = NC * lo_pairs;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gstride = (int64_t)blockDim.x * gridDim.x;

    const int lo_interior_begin = 4;
    const int lo_interior_end_inclusive = Lin - 4; // same as prior

    for (; tid < total_pairs; tid += gstride) {
        int64_t t = tid;
        int64_t lop = t % lo_pairs;
        int64_t nc = t / lo_pairs;

        int c = (int)(nc % C);
        int n = (int)(nc / C);

        int lo0 = (int)(lop * 2);
        int lo1 = lo0 + 1;

        const int64_t xbase = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Lin;
        const int64_t ybase = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Lout;

        // compute output for lo0
        if (lo0 < Lout) {
            float sum0 = 0.0f;
            const bool interior0 =
                ((unsigned)(lo0 - lo_interior_begin) <= (unsigned)(lo_interior_end_inclusive - lo_interior_begin));

            if (interior0) {
                const int lstart0 = lo0 - 4;
                const float* xp0 = x + xbase + (int64_t)lstart0;
                // vectorized if 16B aligned
                if ((((uintptr_t)xp0) & 15) == 0) {
                    const float4* x4 = reinterpret_cast<const float4*>(xp0);
                    float4 v0 = x4[0];
                    float4 v1 = x4[1];
                    sum0 = (v0.x + v0.y + v0.z + v0.w) + (v1.x + v1.y + v1.z + v1.w);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) sum0 += ldg_f32(xp0 + i);
                }
            } else {
                int lstart = lo0 - 4;
                int lend = lstart + 8;
                int l0 = lstart < 0 ? 0 : lstart;
                int l1 = lend > Lin ? Lin : lend;
                #pragma unroll
                for (int li = l0; li < l1; ++li) sum0 += ldg_f32(x + xbase + (int64_t)li);
            }
            y[ybase + (int64_t)lo0] = sum0 * 0.125f;
        }

        // compute output for lo1 (independent stream -> ILP)
        if (lo1 < Lout) {
            float sum1 = 0.0f;
            const bool interior1 =
                ((unsigned)(lo1 - lo_interior_begin) <= (unsigned)(lo_interior_end_inclusive - lo_interior_begin));

            if (interior1) {
                const int lstart1 = lo1 - 4;
                const float* xp1 = x + xbase + (int64_t)lstart1;
                if ((((uintptr_t)xp1) & 15) == 0) {
                    const float4* x4 = reinterpret_cast<const float4*>(xp1);
                    float4 v0 = x4[0];
                    float4 v1 = x4[1];
                    sum1 = (v0.x + v0.y + v0.z + v0.w) + (v1.x + v1.y + v1.z + v1.w);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) sum1 += ldg_f32(xp1 + i);
                }
            } else {
                int lstart = lo1 - 4;
                int lend = lstart + 8;
                int l0 = lstart < 0 ? 0 : lstart;
                int l1 = lend > Lin ? Lin : lend;
                #pragma unroll
                for (int li = l0; li < l1; ++li) sum1 += ldg_f32(x + xbase + (int64_t)li);
            }
            y[ybase + (int64_t)lo1] = sum1 * 0.125f;
        }
    }
}

// -------------------- Generic stride==1, arbitrary k,p --------------------
__global__ __launch_bounds__(256, 2)
void avg_pool1d_forward_stride1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total, int C, int Lin, int Lout, int kL, int pL
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gstride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < (int64_t)total; idx += gstride) {
        int lo = (int)(idx % Lout);
        int tmp = (int)(idx / Lout);
        int c = tmp % C;
        int n = tmp / C;

        int lstart = lo - pL;
        int lend = lstart + kL;
        int l0 = lstart < 0 ? 0 : lstart;
        int l1 = lend > Lin ? Lin : lend;

        const int64_t xbase = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)Lin;
        float sum = 0.0f;

        int li = l0;
        #pragma unroll 4
        for (; li + 3 < l1; li += 4) {
            sum += ldg_f32(x + xbase + (int64_t)(li + 0));
            sum += ldg_f32(x + xbase + (int64_t)(li + 1));
            sum += ldg_f32(x + xbase + (int64_t)(li + 2));
            sum += ldg_f32(x + xbase + (int64_t)(li + 3));
        }
        for (; li < l1; ++li) sum += ldg_f32(x + xbase + (int64_t)li);

        y[idx] = sum / (float)kL; // count_include_pad=True
    }
}

static inline int get_sm_count() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return prop.multiProcessorCount;
}

torch::Tensor avg_pool1d_forward_cuda(torch::Tensor x, int64_t kL, int64_t sL, int64_t pL) {
    CHECK_CUDA(x);
    CHECK_DTYPE(x);
    TORCH_CHECK(x.dim() == 3, "avg_pool1d_forward_cuda: expected input of shape (N, C, L)");
    TORCH_CHECK(kL > 0, "kernel_size must be > 0");
    TORCH_CHECK(sL > 0, "stride must be > 0");
    TORCH_CHECK(pL >= 0, "padding must be >= 0");

    auto x_contig = x.contiguous();

    const int64_t N64 = x_contig.size(0);
    const int64_t C64 = x_contig.size(1);
    const int64_t Lin64 = x_contig.size(2);

    TORCH_CHECK(N64 <= INT_MAX && C64 <= INT_MAX && Lin64 <= INT_MAX, "tensor too large");
    int N = (int)N64;
    int C = (int)C64;
    int Lin = (int)Lin64;

    const int64_t Lout64 = (Lin64 + 2 * pL - kL) / sL + 1;
    TORCH_CHECK(Lout64 >= 0, "avg_pool1d_forward_cuda: computed output length is negative");
    TORCH_CHECK(Lout64 <= INT_MAX, "Lout too large");
    int Lout = (int)Lout64;

    auto y = torch::empty({N64, C64, Lout64}, x_contig.options());

    const int threads = 256;
    const int64_t total64 = N64 * C64 * Lout64;
    TORCH_CHECK(total64 <= INT_MAX, "total elements too large");
    int total = (int)total64;

    int sm = get_sm_count();
    int blocks = (int)div_up_int64_host(total64, threads);
    int target = sm * 20;
    if (blocks > target) blocks = target;
    if (blocks < sm * 2) blocks = sm * 2;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (sL == 1 && kL == 8 && pL == 4) {
        // Tune blocks for pair-work: more CTAs to cover latency (still capped).
        int blocks2 = blocks;
        int target2 = sm * 28;
        if (blocks2 > target2) blocks2 = target2;
        if (blocks2 < sm * 4) blocks2 = sm * 4;

        avg_pool1d_forward_k8s1p4_kernel_ilp2<<<blocks2, threads, 0, stream>>>(
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, Lin, Lout
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }

    if (sL == 1) {
        avg_pool1d_forward_stride1_kernel<<<blocks, threads, 0, stream>>>(
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            total, C, Lin, Lout, (int)kL, (int)pL
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }

    {
        int blocks3 = (int)div_up_int64_host(total64, threads);
        avg_pool1d_forward_baseline_kernel<<<blocks3, threads, 0, stream>>>(
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, Lin, Lout,
            (int)kL, (int)sL, (int)pL
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}
"""

avg_pool1d_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor avg_pool1d_forward_cuda(torch::Tensor x, int64_t kL, int64_t sL, int64_t pL);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_avgpool1d_v8_ilp2_k8s1p4",
    cpp_sources=avg_pool1d_cpp_src,
    cuda_sources=avg_pool1d_cuda_src,
    functions=["avg_pool1d_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--ptxas-options=-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Replacement model using a custom CUDA kernel for AvgPool1d forward.
    Matches nn.AvgPool1d defaults (ceil_mode=False, count_include_pad=True).
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kL = int(kernel_size)
        self.sL = int(stride)
        self.pL = int(padding)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 3):
            return F.avg_pool1d(
                x,
                kernel_size=self.kL,
                stride=self.sL,
                padding=self.pL,
                ceil_mode=False,
                count_include_pad=True,
            )
        return self.custom_ops.avg_pool1d_forward_cuda(x, self.kL, self.sL, self.pL)