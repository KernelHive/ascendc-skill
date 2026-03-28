import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

pam_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ bool is_aligned_16(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0);
}

#if __CUDA_ARCH__ >= 350
static __device__ __forceinline__ float ldg_f32(const float* p) {
    return __ldg(p);
}
#else
static __device__ __forceinline__ float ldg_f32(const float* p) {
    return *p;
}
#endif

// HW==49 fast path: unrolled q loop.
// Mapping: grid.x = N * HW (one block per (n,p))
// Threads cover C in float4 packs: pack = threadIdx.x + blockDim.x * blockIdx.y (blockIdx.y tiles packs).
// This supports any C divisible by 4 (dominant C=512).
__global__ void pam_tail_np_cpack_hw49_kernel(
    const float* __restrict__ x,     // [N,C,HW] view of NCHW contiguous
    const float* __restrict__ attn,  // [N,HW,HW]
    const float* __restrict__ D,     // [N,HW,C]
    float* __restrict__ out,         // [N,C,HW]
    int N, int C, int HW,
    float alpha
) {
    int np = (int)blockIdx.x;
    int n = np / HW;
    int p = np - n * HW;
    if (n >= N) return;

    // Require HW==49 for this kernel.
    // C pack
    int C4 = C >> 2;

    int pack = (int)threadIdx.x + (int)blockDim.x * (int)blockIdx.y;
    if (pack >= C4) return;

    // Stage attn row into shared memory: 49 floats.
    extern __shared__ float s_attn[];
    // Cooperative load
    for (int q = (int)threadIdx.x; q < 49; q += (int)blockDim.x) {
        s_attn[q] = ldg_f32(attn + ((int64_t)n * 49 + p) * 49 + q);
    }
    __syncthreads();

    const bool vec_ok = (C % 4 == 0) && is_aligned_16(D) && is_aligned_16(x) && is_aligned_16(out);

    // Base pointers
    const float* Dn = D + (int64_t)n * 49 * (int64_t)C;
    const float* xn = x + (int64_t)n * (int64_t)C * 49;
    float* on = out + (int64_t)n * (int64_t)C * 49;

    float4 acc = {0.f, 0.f, 0.f, 0.f};

    if (vec_ok) {
        const float4* __restrict__ D4 = reinterpret_cast<const float4*>(Dn);
        // Stream through q with pointer increments to avoid expensive mul in loop.
        const float4* dptr = D4 + pack; // q=0
        #pragma unroll
        for (int q = 0; q < 49; ++q) {
            float a = s_attn[q];
            float4 dv = ldg_f32(reinterpret_cast<const float*>(dptr) + 0) ? *dptr : *dptr; // keep compiler from being too clever
            // NOTE: above line effectively is *dptr; ldg for float4 not directly used.
            acc.x = fmaf(a, dv.x, acc.x);
            acc.y = fmaf(a, dv.y, acc.y);
            acc.z = fmaf(a, dv.z, acc.z);
            acc.w = fmaf(a, dv.w, acc.w);
            dptr += C4;
        }

        const float4* __restrict__ X4 = reinterpret_cast<const float4*>(xn);
        float4* __restrict__ O4 = reinterpret_cast<float4*>(on);
        int64_t idx4 = ((int64_t)pack) * 49 + p; // within batch n
        float4 xv = X4[idx4];
        float4 ov;
        ov.x = fmaf(alpha, acc.x, xv.x);
        ov.y = fmaf(alpha, acc.y, xv.y);
        ov.z = fmaf(alpha, acc.z, xv.z);
        ov.w = fmaf(alpha, acc.w, xv.w);
        O4[idx4] = ov;
    } else {
        // Scalar path (still staged attn row).
        int c0 = pack * 4;
        const float* dptr = Dn + c0; // q=0
        #pragma unroll
        for (int q = 0; q < 49; ++q) {
            float a = s_attn[q];
            acc.x = fmaf(a, dptr[0], acc.x);
            acc.y = fmaf(a, dptr[1], acc.y);
            acc.z = fmaf(a, dptr[2], acc.z);
            acc.w = fmaf(a, dptr[3], acc.w);
            dptr += C;
        }
        int64_t base = ((int64_t)c0) * 49 + p; // within batch n
        on[base + 0 * 49] = fmaf(alpha, acc.x, xn[base + 0 * 49]);
        on[base + 1 * 49] = fmaf(alpha, acc.y, xn[base + 1 * 49]);
        on[base + 2 * 49] = fmaf(alpha, acc.z, xn[base + 2 * 49]);
        on[base + 3 * 49] = fmaf(alpha, acc.w, xn[base + 3 * 49]);
    }
}

// Generic HW kernel (still (n,p) blocks + pack tiling, but loop over HW).
__global__ void pam_tail_np_cpack_generic_kernel(
    const float* __restrict__ x,     // [N,C,HW]
    const float* __restrict__ attn,  // [N,HW,HW]
    const float* __restrict__ D,     // [N,HW,C]
    float* __restrict__ out,         // [N,C,HW]
    int N, int C, int HW,
    float alpha
) {
    int np = (int)blockIdx.x;
    int n = np / HW;
    int p = np - n * HW;
    if (n >= N) return;

    int C4 = C >> 2;
    int pack = (int)threadIdx.x + (int)blockDim.x * (int)blockIdx.y;
    if (pack >= C4) return;

    extern __shared__ float s_attn[];
    const int64_t attn_row_base = ((int64_t)n * HW + p) * (int64_t)HW;

    for (int q = (int)threadIdx.x; q < HW; q += (int)blockDim.x) {
        s_attn[q] = ldg_f32(attn + attn_row_base + q);
    }
    __syncthreads();

    const bool vec_ok = (C % 4 == 0) && is_aligned_16(D) && is_aligned_16(x) && is_aligned_16(out);

    const float* Dn = D + (int64_t)n * (int64_t)HW * (int64_t)C;
    const float* xn = x + (int64_t)n * (int64_t)C * (int64_t)HW;
    float* on = out + (int64_t)n * (int64_t)C * (int64_t)HW;

    float4 acc = {0.f, 0.f, 0.f, 0.f};

    if (vec_ok) {
        const float4* __restrict__ D4 = reinterpret_cast<const float4*>(Dn);
        const float4* dptr = D4 + pack;
        for (int q = 0; q < HW; ++q) {
            float a = s_attn[q];
            float4 dv = *dptr;
            acc.x = fmaf(a, dv.x, acc.x);
            acc.y = fmaf(a, dv.y, acc.y);
            acc.z = fmaf(a, dv.z, acc.z);
            acc.w = fmaf(a, dv.w, acc.w);
            dptr += C4;
        }

        const float4* __restrict__ X4 = reinterpret_cast<const float4*>(xn);
        float4* __restrict__ O4 = reinterpret_cast<float4*>(on);
        int64_t idx4 = ((int64_t)pack) * (int64_t)HW + p;
        float4 xv = X4[idx4];
        float4 ov;
        ov.x = fmaf(alpha, acc.x, xv.x);
        ov.y = fmaf(alpha, acc.y, xv.y);
        ov.z = fmaf(alpha, acc.z, xv.z);
        ov.w = fmaf(alpha, acc.w, xv.w);
        O4[idx4] = ov;
    } else {
        int c0 = pack * 4;
        const float* dptr = Dn + c0;
        for (int q = 0; q < HW; ++q) {
            float a = s_attn[q];
            acc.x = fmaf(a, dptr[0], acc.x);
            acc.y = fmaf(a, dptr[1], acc.y);
            acc.z = fmaf(a, dptr[2], acc.z);
            acc.w = fmaf(a, dptr[3], acc.w);
            dptr += C;
        }
        int64_t base = ((int64_t)c0) * (int64_t)HW + p;
        on[base + 0 * (int64_t)HW] = fmaf(alpha, acc.x, xn[base + 0 * (int64_t)HW]);
        on[base + 1 * (int64_t)HW] = fmaf(alpha, acc.y, xn[base + 1 * (int64_t)HW]);
        on[base + 2 * (int64_t)HW] = fmaf(alpha, acc.z, xn[base + 2 * (int64_t)HW]);
        on[base + 3 * (int64_t)HW] = fmaf(alpha, acc.w, xn[base + 3 * (int64_t)HW]);
    }
}

torch::Tensor pam_tail_fused_cuda(torch::Tensor x, torch::Tensor attn, torch::Tensor D, torch::Tensor alpha_t) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(attn.is_cuda(), "attn must be a CUDA tensor");
    TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(attn.dtype() == torch::kFloat32, "attn must be float32");
    TORCH_CHECK(D.dtype() == torch::kFloat32, "D must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(attn.is_contiguous(), "attn must be contiguous");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be (N,C,H,W)");
    TORCH_CHECK(attn.dim() == 3, "attn must be (N,HW,HW)");
    TORCH_CHECK(D.dim() == 3, "D must be (N,HW,C)");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;

    TORCH_CHECK(attn.size(0) == N, "attn N mismatch");
    TORCH_CHECK(attn.size(1) == HW && attn.size(2) == HW, "attn must be (N,HW,HW) with HW=H*W");
    TORCH_CHECK(D.size(0) == N, "D N mismatch");
    TORCH_CHECK(D.size(1) == HW, "D must be (N,HW,C) with HW=H*W");
    TORCH_CHECK(D.size(2) == C, "D C mismatch");

    TORCH_CHECK(alpha_t.numel() == 1, "alpha must have 1 element");
    TORCH_CHECK(alpha_t.dtype() == torch::kFloat32, "alpha must be float32");

    float alpha = alpha_t.item<float>();

    auto out = torch::empty_like(x);

    TORCH_CHECK((C % 4) == 0, "C must be divisible by 4 for this optimized kernel");

    // Grid mapping:
    // blocks_x = N*HW (one per (n,p))
    // blocks_y = ceil(C4 / packs_per_block)
    // threads = packs_per_block
    const int C4 = C >> 2;

    // Use 128 threads by default: good balance for C4=128 (C=512) => blocks_y=1.
    // If C is smaller, still fine; if larger, blocks_y>1.
    const int threads = 128;
    const int blocks_x = N * HW;
    const int blocks_y = (C4 + threads - 1) / threads;

    dim3 grid(blocks_x, blocks_y, 1);
    dim3 block(threads, 1, 1);

    // Shared memory: stage attn row.
    size_t shmem = (size_t)HW * sizeof(float);

    if (HW == 49) {
        // For HW==49, allocate exactly 49 floats.
        shmem = 49 * sizeof(float);
        pam_tail_np_cpack_hw49_kernel<<<grid, block, shmem>>>(
            x.data_ptr<float>(),
            attn.data_ptr<float>(),
            D.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, HW,
            alpha
        );
    } else {
        pam_tail_np_cpack_generic_kernel<<<grid, block, shmem>>>(
            x.data_ptr<float>(),
            attn.data_ptr<float>(),
            D.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, HW,
            alpha
        );
    }

    return out;
}
"""

pam_cpp_src = r"""
torch::Tensor pam_tail_fused_cuda(torch::Tensor x, torch::Tensor attn, torch::Tensor D, torch::Tensor alpha_t);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_pam_v3",
    cpp_sources=pam_cpp_src,
    cuda_sources=pam_cuda_src,
    functions=["pam_tail_fused_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Position Attention Module (PAM) with an optimized fused CUDA tail:
      out = x + alpha * reshape((attn @ D).transpose(1,2), [N,C,H,W])
    """
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Conv2d(dim, dim, 1)
        self.c = nn.Conv2d(dim, dim, 1)
        self.d = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        n, c, h, w = x.shape

        B = self.b(x).flatten(2).transpose(1, 2)   # (N, HW, C)
        Cq = self.c(x).flatten(2)                  # (N, C, HW)
        D = self.d(x).flatten(2).transpose(1, 2)   # (N, HW, C)

        attn = (B @ Cq).softmax(dim=-1)            # (N, HW, HW)

        if x.is_cuda and x.dtype == torch.float32 and (c % 4 == 0):
            return self.custom_ops.pam_tail_fused_cuda(
                x.contiguous(),
                attn.contiguous(),
                D.contiguous(),
                self.alpha,
            )

        y = (attn @ D).transpose(1, 2).reshape(n, c, h, w)
        return self.alpha * y + x


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512]