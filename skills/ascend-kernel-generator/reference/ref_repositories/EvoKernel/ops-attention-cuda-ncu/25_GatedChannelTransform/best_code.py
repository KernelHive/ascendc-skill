import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ extension: optimized GCT forward (mode='l2' only)
gct_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Kernel A (fast path): warp-per-(n,c) reduction for HW<=64.
// Also computes embedding[n,c] = sqrt(sumsq+eps) * alpha[c] and stores it.
__global__ void gct_l2_sumsq_embed_warp_kernel(
    const float* __restrict__ x,      // [NC*HW]
    const float* __restrict__ alpha,  // [C]
    float* __restrict__ sumsq_nc,     // [NC]
    float* __restrict__ embed_nc,     // [NC]
    int NC, int C, int HW,
    float eps
){
    int warps_per_block = (blockDim.x >> 5);
    int warp_id = (threadIdx.x >> 5);
    int lane = (threadIdx.x & 31);
    int warp_global = blockIdx.x * warps_per_block + warp_id;

    int nc = warp_global;
    if (nc >= NC) return;

    int c = nc % C;

    const float* xptr = x + (size_t)nc * (size_t)HW;

    float acc = 0.0f;
    // lane-stride loads; HW<=64 typical 49
    for (int i = lane; i < HW; i += 32) {
        float v = ldg_f32(xptr + i);
        acc = fmaf(v, v, acc);
    }
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        float a = ldg_f32(alpha + c);
        float emb = sqrtf(acc + eps) * a;
        sumsq_nc[nc] = acc;
        embed_nc[nc] = emb;
    }
}

// Fallback Kernel A: one block per (n,c) reduce over HW (generic HW).
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    float out = 0.0f;
    if (wid == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        out = (lane < nwarps) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

__global__ void gct_l2_sumsq_embed_block_kernel(
    const float* __restrict__ x,      // [NC*HW]
    const float* __restrict__ alpha,  // [C]
    float* __restrict__ sumsq_nc,     // [NC]
    float* __restrict__ embed_nc,     // [NC]
    int NC, int C, int HW,
    float eps
){
    int nc = blockIdx.x;
    if (nc >= NC) return;
    int c = nc % C;

    const float* xptr = x + (size_t)nc * (size_t)HW;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float v = ldg_f32(xptr + i);
        acc = fmaf(v, v, acc);
    }
    float total = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        float a = ldg_f32(alpha + c);
        float emb = sqrtf(total + eps) * a;
        sumsq_nc[nc] = total;
        embed_nc[nc] = emb;
    }
}

// Kernel B: per-n mean of embed^2, output inv_sqrt_mean[n] = rsqrt(mean(embed^2)+eps)
__global__ void gct_l2_inv_sqrt_mean_kernel(
    const float* __restrict__ embed_nc,   // [N*C]
    float* __restrict__ inv_sqrt_mean,    // [N]
    int N, int C,
    float eps
){
    int n = blockIdx.x;
    if (n >= N) return;

    float acc = 0.0f;
    int base = n * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float e = ldg_f32(embed_nc + base + c);
        acc += e * e;
    }
    float total = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        float mean = total * (1.0f / (float)C);
        inv_sqrt_mean[n] = rsqrtf(mean + eps);
    }
}

// Kernel C: apply gate, one thread per element; optional float4 vectorization over flattened tensor.
__global__ void gct_l2_apply_elem_kernel(
    const float* __restrict__ x,             // [N*C*HW]
    const float* __restrict__ embed_nc,      // [N*C]
    const float* __restrict__ inv_sqrt_mean, // [N]
    const float* __restrict__ gamma,         // [C]
    const float* __restrict__ beta,          // [C]
    float* __restrict__ out,                 // [N*C*HW]
    int N, int C, int HW
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx >= total) return;

    int t = idx / HW;
    int c = t % C;
    int n = t / C;
    int nc = n * C + c;

    float emb = ldg_f32(embed_nc + nc);
    float inv = ldg_f32(inv_sqrt_mean + n);
    float g = ldg_f32(gamma + c);
    float be = ldg_f32(beta + c);

    float z = fmaf(emb, g * inv, be);
    float gate = 1.0f + tanhf(z);

    out[idx] = x[idx] * gate;
}

// Vectorized apply (float4) over the flattened tensor for stable alignment.
// Each thread processes one float4. Handles tail separately.
__global__ void gct_l2_apply_elem_kernel_vec4(
    const float4* __restrict__ x4,           // [total4]
    const float* __restrict__ embed_nc,      // [N*C]
    const float* __restrict__ inv_sqrt_mean, // [N]
    const float* __restrict__ gamma,         // [C]
    const float* __restrict__ beta,          // [C]
    float4* __restrict__ out4,               // [total4]
    int N, int C, int HW, int total4
){
    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= total4) return;

    int base_idx = i4 * 4; // element index
    int t = base_idx / HW;
    int c = t % C;
    int n = t / C;
    int nc = n * C + c;

    float emb = ldg_f32(embed_nc + nc);
    float inv = ldg_f32(inv_sqrt_mean + n);
    float g = ldg_f32(gamma + c);
    float be = ldg_f32(beta + c);
    float z = fmaf(emb, g * inv, be);
    float gate = 1.0f + tanhf(z);

    float4 v = x4[i4];
    v.x *= gate; v.y *= gate; v.z *= gate; v.w *= gate;
    out4[i4] = v;
}

torch::Tensor gct_forward_cuda(
    torch::Tensor x,
    torch::Tensor alpha,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon
){
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(alpha.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "params must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(alpha.dtype() == torch::kFloat32 && gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32,
                "params must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    auto x_contig = x.contiguous();
    int N = (int)x_contig.size(0);
    int C = (int)x_contig.size(1);
    int H = (int)x_contig.size(2);
    int W = (int)x_contig.size(3);
    int HW = H * W;
    int NC = N * C;

    // Flatten params to [C]
    auto a = alpha.contiguous().view({C});
    auto g = gamma.contiguous().view({C});
    auto b = beta.contiguous().view({C});

    auto sumsq_nc = torch::empty({NC}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    auto embed_nc = torch::empty({NC}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    auto inv_sqrt_mean = torch::empty({N}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    auto out = torch::empty_like(x_contig);

    float eps = (float)epsilon;

    // Kernel A: pick warp fast-path for HW<=64
    if (HW <= 64) {
        int threads = 256; // 8 warps
        int warps_per_block = threads / 32;
        int blocks = (NC + warps_per_block - 1) / warps_per_block;
        gct_l2_sumsq_embed_warp_kernel<<<blocks, threads>>>(
            (const float*)x_contig.data_ptr<float>(),
            (const float*)a.data_ptr<float>(),
            (float*)sumsq_nc.data_ptr<float>(),
            (float*)embed_nc.data_ptr<float>(),
            NC, C, HW, eps
        );
    } else {
        int threads = 256;
        gct_l2_sumsq_embed_block_kernel<<<NC, threads>>>(
            (const float*)x_contig.data_ptr<float>(),
            (const float*)a.data_ptr<float>(),
            (float*)sumsq_nc.data_ptr<float>(),
            (float*)embed_nc.data_ptr<float>(),
            NC, C, HW, eps
        );
    }

    // Kernel B: per-n inv sqrt mean
    int threadsB = 256;
    gct_l2_inv_sqrt_mean_kernel<<<N, threadsB>>>(
        (const float*)embed_nc.data_ptr<float>(),
        (float*)inv_sqrt_mean.data_ptr<float>(),
        N, C, eps
    );

    // Kernel C: apply, try vec4 over flattened tensor if aligned and size permits.
    int total = N * C * HW;
    const float* xptr = (const float*)x_contig.data_ptr<float>();
    float* optr = (float*)out.data_ptr<float>();

    bool aligned16 = (((uintptr_t)xptr & 0xF) == 0) && (((uintptr_t)optr & 0xF) == 0);
    int total4 = total / 4;

    if (aligned16 && total4 > 0) {
        int threads = 256;
        int blocks = (total4 + threads - 1) / threads;
        gct_l2_apply_elem_kernel_vec4<<<blocks, threads>>>(
            (const float4*)xptr,
            (const float*)embed_nc.data_ptr<float>(),
            (const float*)inv_sqrt_mean.data_ptr<float>(),
            (const float*)g.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (float4*)optr,
            N, C, HW, total4
        );
        int tail = total - total4 * 4;
        if (tail) {
            int start = total4 * 4;
            int threads2 = 256;
            int blocks2 = (tail + threads2 - 1) / threads2;
            gct_l2_apply_elem_kernel<<<blocks2, threads2>>>(
                xptr + start,
                (const float*)embed_nc.data_ptr<float>(),
                (const float*)inv_sqrt_mean.data_ptr<float>(),
                (const float*)g.data_ptr<float>(),
                (const float*)b.data_ptr<float>(),
                optr + start,
                N, C, HW
            );
        }
    } else {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        gct_l2_apply_elem_kernel<<<blocks, threads>>>(
            xptr,
            (const float*)embed_nc.data_ptr<float>(),
            (const float*)inv_sqrt_mean.data_ptr<float>(),
            (const float*)g.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            optr,
            N, C, HW
        );
    }

    return out;
}
"""

gct_cpp_source = r"""
torch::Tensor gct_forward_cuda(
    torch::Tensor x,
    torch::Tensor alpha,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon
);
"""

custom_ops_lib = load_inline(
    name="custom_gct_ops_opt7",
    cpp_sources=gct_cpp_source,
    cuda_sources=gct_cuda_source,
    functions=["gct_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode="l2", after_relu=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = float(epsilon)
        self.mode = mode
        self.after_relu = after_relu
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        if self.mode != "l2":
            raise NotImplementedError("Custom CUDA kernel implemented for mode='l2' only.")
        if x.dtype != torch.float32:
            x = x.float()
        return self.custom_ops_lib.gct_forward_cuda(x, self.alpha, self.gamma, self.beta, self.epsilon)