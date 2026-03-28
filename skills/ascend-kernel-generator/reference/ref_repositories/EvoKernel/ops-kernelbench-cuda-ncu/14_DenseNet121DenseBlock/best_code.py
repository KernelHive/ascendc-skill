import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: fused concat along channel for NCHW float32 tensors
# Single-pass write to output with vectorized IO fast paths.
# ------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static inline __host__ __device__ bool is_aligned_16(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xFULL) == 0ULL);
}
static inline __host__ __device__ bool is_aligned_8(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0x7ULL) == 0ULL);
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void concat_nchw_fused_kernel_scalar(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C1, int C2, int H, int W
) {
    const int C = C1 + C2;
    const long long HW = (long long)H * (long long)W;
    const long long total = (long long)N * (long long)C * HW;

    long long idx = (long long)blockIdx.x * (long long)THREADS + (long long)threadIdx.x;
    long long stride = (long long)gridDim.x * (long long)THREADS;

    for (; idx < total; idx += stride) {
        // idx maps to (((n*C + c)*H + h)*W + w) in NCHW contiguous
        long long t = idx;
        long long hw = t % HW; t /= HW;
        int c = (int)(t % (long long)C);
        int n = (int)(t / (long long)C);

        if (c < C1) {
            y[idx] = a[((long long)n * C1 + c) * HW + hw];
        } else {
            int cb = c - C1;
            y[idx] = b[((long long)n * C2 + cb) * HW + hw];
        }
    }
}

// Vectorized float4 variant: operates on float4 elements of the output.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void concat_nchw_fused_kernel_float4(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C1, int C2, int H, int W
) {
    const int C = C1 + C2;
    const long long HW = (long long)H * (long long)W;
    const long long total = (long long)N * (long long)C * HW;
    const long long total4 = total >> 2; // number of float4s

    const float4* __restrict__ a4 = reinterpret_cast<const float4*>(a);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(b);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    long long i = (long long)blockIdx.x * (long long)THREADS + (long long)threadIdx.x;
    long long stride = (long long)gridDim.x * (long long)THREADS;

    // Each i corresponds to 4 consecutive floats in flattened NCHW.
    for (; i < total4; i += stride) {
        long long idx = i << 2; // base scalar index
        // Determine mapping using scalar base index (safe because 4-wide stays within same contiguous region).
        long long t = idx;
        long long hw = t % HW; t /= HW;
        int c = (int)(t % (long long)C);
        int n = (int)(t / (long long)C);

        // Fast path: if hw is multiple of 4, then these 4 floats stay within same plane.
        // If not, they may cross a plane boundary; fall back to scalar for correctness.
        if ((hw & 3LL) == 0LL) {
            if (c < C1) {
                long long a_idx = (((long long)n * C1 + c) * HW + hw) >> 2;
                y4[i] = a4[a_idx];
            } else {
                int cb = c - C1;
                long long b_idx = (((long long)n * C2 + cb) * HW + hw) >> 2;
                y4[i] = b4[b_idx];
            }
        } else {
            // Scalar fallback for this vector to avoid cross-boundary issues.
            float4 out;
            float tmp[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                long long idxk = idx + k;
                long long tt = idxk;
                long long hwk = tt % HW; tt /= HW;
                int ck = (int)(tt % (long long)C);
                int nk = (int)(tt / (long long)C);
                if (ck < C1) tmp[k] = a[((long long)nk * C1 + ck) * HW + hwk];
                else         tmp[k] = b[((long long)nk * C2 + (ck - C1)) * HW + hwk];
            }
            out.x = tmp[0]; out.y = tmp[1]; out.z = tmp[2]; out.w = tmp[3];
            y4[i] = out;
        }
    }
}

// Vectorized float2 variant: similar logic to float4.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void concat_nchw_fused_kernel_float2(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C1, int C2, int H, int W
) {
    const int C = C1 + C2;
    const long long HW = (long long)H * (long long)W;
    const long long total = (long long)N * (long long)C * HW;
    const long long total2 = total >> 1;

    const float2* __restrict__ a2 = reinterpret_cast<const float2*>(a);
    const float2* __restrict__ b2 = reinterpret_cast<const float2*>(b);
    float2* __restrict__ y2 = reinterpret_cast<float2*>(y);

    long long i = (long long)blockIdx.x * (long long)THREADS + (long long)threadIdx.x;
    long long stride = (long long)gridDim.x * (long long)THREADS;

    for (; i < total2; i += stride) {
        long long idx = i << 1;
        long long t = idx;
        long long hw = t % HW; t /= HW;
        int c = (int)(t % (long long)C);
        int n = (int)(t / (long long)C);

        if ((hw & 1LL) == 0LL) {
            if (c < C1) {
                long long a_idx = (((long long)n * C1 + c) * HW + hw) >> 1;
                y2[i] = a2[a_idx];
            } else {
                int cb = c - C1;
                long long b_idx = (((long long)n * C2 + cb) * HW + hw) >> 1;
                y2[i] = b2[b_idx];
            }
        } else {
            float2 out;
            float tmp0, tmp1;
            long long idx0 = idx;
            long long t0 = idx0;
            long long hw0 = t0 % HW; t0 /= HW;
            int c0 = (int)(t0 % (long long)C);
            int n0 = (int)(t0 / (long long)C);

            long long idx1 = idx + 1;
            long long t1 = idx1;
            long long hw1 = t1 % HW; t1 /= HW;
            int c1v = (int)(t1 % (long long)C);
            int n1 = (int)(t1 / (long long)C);

            tmp0 = (c0 < C1) ? a[((long long)n0 * C1 + c0) * HW + hw0]
                             : b[((long long)n0 * C2 + (c0 - C1)) * HW + hw0];
            tmp1 = (c1v < C1) ? a[((long long)n1 * C1 + c1v) * HW + hw1]
                              : b[((long long)n1 * C2 + (c1v - C1)) * HW + hw1];

            out.x = tmp0; out.y = tmp1;
            y2[i] = out;
        }
    }
}

torch::Tensor concat_channel_nchw_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_CUDA(a); CHECK_CUDA(b);
    CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b);
    CHECK_FLOAT(a); CHECK_FLOAT(b);

    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "a and b must be NCHW tensors");
    TORCH_CHECK(a.size(0) == b.size(0), "N mismatch");
    TORCH_CHECK(a.size(2) == b.size(2) && a.size(3) == b.size(3), "H/W mismatch");

    auto N64 = a.size(0);
    auto C1_64 = a.size(1);
    auto C2_64 = b.size(1);
    auto H64 = a.size(2);
    auto W64 = a.size(3);

    TORCH_CHECK(N64 <= INT_MAX && C1_64 <= INT_MAX && C2_64 <= INT_MAX && H64 <= INT_MAX && W64 <= INT_MAX,
               "sizes too large");

    int N = (int)N64;
    int C1 = (int)C1_64;
    int C2 = (int)C2_64;
    int H = (int)H64;
    int W = (int)W64;
    int C = C1 + C2;

    c10::cuda::CUDAGuard device_guard(a.device());
    auto y = torch::empty({N64, (int64_t)C, H64, W64}, a.options());

    const float* ap = a.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    float* yp = y.data_ptr<float>();

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const long long total = (long long)N * (long long)C * (long long)H * (long long)W;

    // Heuristic: enough blocks to cover SMs; keep it moderate to reduce launch overhead.
    // Use 128 threads to reduce per-thread regs and increase residency.
    const int threads = 128;
    int blocks = (int)((total + threads - 1LL) / threads);
    // Cap blocks to avoid excessive launch overhead; still enough for occupancy.
    blocks = blocks > 65535 ? 65535 : blocks;
    blocks = blocks < 80 ? 80 : blocks; // ensure some parallelism (typical SM count order)

    bool can4 = ((total & 3LL) == 0LL) && is_aligned_16(ap) && is_aligned_16(bp) && is_aligned_16(yp);
    bool can2 = ((total & 1LL) == 0LL) && is_aligned_8(ap) && is_aligned_8(bp) && is_aligned_8(yp);

    if (can4) {
        concat_nchw_fused_kernel_float4<threads><<<blocks, threads, 0, stream>>>(ap, bp, yp, N, C1, C2, H, W);
    } else if (can2) {
        concat_nchw_fused_kernel_float2<threads><<<blocks, threads, 0, stream>>>(ap, bp, yp, N, C1, C2, H, W);
    } else {
        concat_nchw_fused_kernel_scalar<threads><<<blocks, threads, 0, stream>>>(ap, bp, yp, N, C1, C2, H, W);
    }

    return y;
}
"""

cpp_src = r"""
torch::Tensor concat_channel_nchw_cuda(torch::Tensor a, torch::Tensor b);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_dense_net121dense_block_concat_v3_fused",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["concat_channel_nchw_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        layers = []
        num_layers = int(num_layers)
        num_input_features = int(num_input_features)
        growth_rate = int(growth_rate)

        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.Dropout(0.0),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        for layer in self.layers:
            new_feature = layer(x).contiguous()
            x = custom_ops_lib.concat_channel_nchw_cuda(x, new_feature)
        return x