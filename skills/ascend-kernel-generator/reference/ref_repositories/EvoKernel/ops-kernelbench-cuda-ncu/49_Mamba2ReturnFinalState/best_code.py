import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float myexp(float x) { return expf(x); }

__device__ __forceinline__ int idx3_bhc(int b, int h, int c, int H, int C1) {
    return ((b * H + h) * C1 + c);
}

// inclusive scan within warp
__device__ __forceinline__ float warp_inclusive_scan(float v, unsigned mask=0xffffffffu) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(mask, v, offset);
        if ((threadIdx.x & 31) >= offset) v += n;
    }
    return v;
}

__device__ __forceinline__ float4 load_f4_unaligned(const float* p) {
    float4 v;
    // Use memcpy-style loads to avoid alignment UB (compiler will generate ld.global.nc as needed)
    v.x = p[0]; v.y = p[1]; v.z = p[2]; v.w = p[3];
    return v;
}
__device__ __forceinline__ void store_f4_unaligned(float* p, const float4& v) {
    p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
}

__global__ void mamba2_return_final_state_vecN16_f32_kernel(
    const float* __restrict__ states_cat, // [B,C1,H,P,N] contiguous
    const float* __restrict__ a_last_pad, // [B,H,C1] contiguous
    float* __restrict__ out,              // [B,H,P,N] contiguous
    int B, int C1, int H, int P
) {
    // blockIdx.y enumerates (b,h)
    int bh = (int)blockIdx.y;
    int b = bh / H;
    int h = bh - b * H;
    if (b >= B) return;

    // Compute p index for this thread (one thread -> one p, computes all N=16)
    int p = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (p >= P) return;

    extern __shared__ float smem[]; // [C1] weights
    float* sw = smem;

    // Build weights sw[c] = exp(total - prefix[c]) where prefix is inclusive cumsum of a_last_pad[b,h,:]
    // Do it with warp0 to minimize overhead; two-pass over tiles of 32 to get total then prefixes.
    if (threadIdx.x < 32) {
        float running = 0.0f;
        // Pass 1: total
        for (int tile = 0; tile < C1; tile += 32) {
            int c = tile + (int)threadIdx.x;
            float v = (c < C1) ? a_last_pad[idx3_bhc(b, h, c, H, C1)] : 0.0f;
            float inc = warp_inclusive_scan(v);
            float tile_sum = __shfl_sync(0xffffffffu, inc, 31);
            running += tile_sum;
        }
        float total = running;

        // Pass 2: prefixes + weights
        running = 0.0f;
        for (int tile = 0; tile < C1; tile += 32) {
            int c = tile + (int)threadIdx.x;
            float v = (c < C1) ? a_last_pad[idx3_bhc(b, h, c, H, C1)] : 0.0f;
            float inc = warp_inclusive_scan(v);
            float prefix_c = running + inc;
            float tile_sum = __shfl_sync(0xffffffffu, inc, 31);
            if (c < C1) sw[c] = myexp(total - prefix_c);
            running += tile_sum;
        }
    }
    __syncthreads();

    // Layout strides (all int32 to reduce overhead; sizes are small in this benchmark)
    // states_cat index: ((((b*C1 + c)*H + h)*P + p)*N + n)
    // With N=16, we treat it as 4 float4s contiguous.
    const int N = 16;
    int stride_c = H * P * N;               // elements to jump when c++
    int stride_h = P * N;                   // within fixed (b,c)
    int stride_b = C1 * H * P * N;

    int base_b = b * stride_b;
    int base_h = h * stride_h;
    // pointer to states_cat[b,0,h,p,0]
    const float* sp = states_cat + base_b + base_h + p * N;

    float4 acc0{0.f, 0.f, 0.f, 0.f};
    float4 acc1{0.f, 0.f, 0.f, 0.f};
    float4 acc2{0.f, 0.f, 0.f, 0.f};
    float4 acc3{0.f, 0.f, 0.f, 0.f};

    // Reduce over c with pointer bumping (no 64-bit indexing in hot loop)
    for (int c = 0; c < C1; ++c) {
        float w = sw[c];
        float4 v0 = load_f4_unaligned(sp + 0);
        float4 v1 = load_f4_unaligned(sp + 4);
        float4 v2 = load_f4_unaligned(sp + 8);
        float4 v3 = load_f4_unaligned(sp + 12);

        acc0.x = fmaf(v0.x, w, acc0.x); acc0.y = fmaf(v0.y, w, acc0.y); acc0.z = fmaf(v0.z, w, acc0.z); acc0.w = fmaf(v0.w, w, acc0.w);
        acc1.x = fmaf(v1.x, w, acc1.x); acc1.y = fmaf(v1.y, w, acc1.y); acc1.z = fmaf(v1.z, w, acc1.z); acc1.w = fmaf(v1.w, w, acc1.w);
        acc2.x = fmaf(v2.x, w, acc2.x); acc2.y = fmaf(v2.y, w, acc2.y); acc2.z = fmaf(v2.z, w, acc2.z); acc2.w = fmaf(v2.w, w, acc2.w);
        acc3.x = fmaf(v3.x, w, acc3.x); acc3.y = fmaf(v3.y, w, acc3.y); acc3.z = fmaf(v3.z, w, acc3.z); acc3.w = fmaf(v3.w, w, acc3.w);

        sp += stride_c;
    }

    // out index: (((b*H + h)*P + p)*N + n)
    int out_stride_bh = P * N;
    int out_base = (b * H + h) * out_stride_bh + p * N;
    float* op = out + out_base;

    store_f4_unaligned(op + 0, acc0);
    store_f4_unaligned(op + 4, acc1);
    store_f4_unaligned(op + 8, acc2);
    store_f4_unaligned(op + 12, acc3);
}

torch::Tensor mamba2_return_final_state_cuda(torch::Tensor states_cat, torch::Tensor a_last_pad) {
    CHECK_CUDA(states_cat);
    CHECK_CUDA(a_last_pad);
    CHECK_CONTIGUOUS(states_cat);
    CHECK_CONTIGUOUS(a_last_pad);
    CHECK_FLOAT(states_cat);
    CHECK_FLOAT(a_last_pad);

    TORCH_CHECK(states_cat.dim() == 5, "states_cat must be 5D [B, C1, H, P, N]");
    TORCH_CHECK(a_last_pad.dim() == 3, "a_last_pad must be 3D [B, H, C1]");

    int64_t B64  = states_cat.size(0);
    int64_t C164 = states_cat.size(1);
    int64_t H64  = states_cat.size(2);
    int64_t P64  = states_cat.size(3);
    int64_t N64  = states_cat.size(4);

    TORCH_CHECK(C164 >= 1, "C1 must be >= 1");
    TORCH_CHECK(N64 == 16, "Optimized kernel supports N==16 only (got N=", N64, ")");
    TORCH_CHECK(a_last_pad.size(0) == B64, "a_last_pad.size(0) must match B");
    TORCH_CHECK(a_last_pad.size(1) == H64, "a_last_pad.size(1) must match H");
    TORCH_CHECK(a_last_pad.size(2) == C164, "a_last_pad.size(2) must match C1");

    auto out = torch::empty({B64, H64, P64, N64}, states_cat.options());
    if (B64 == 0 || H64 == 0 || P64 == 0) return out;

    const float* sptr = (const float*)states_cat.data_ptr<float>();
    const float* aptr = (const float*)a_last_pad.data_ptr<float>();
    float* optr = (float*)out.data_ptr<float>();

    int threads = 256;
    int blocks_x = (int)((P64 + threads - 1) / threads);
    if (blocks_x < 1) blocks_x = 1;

    dim3 grid((unsigned)blocks_x, (unsigned)(B64 * H64), 1);
    dim3 block((unsigned)threads, 1, 1);

    size_t shmem = (size_t)C164 * sizeof(float);
    mamba2_return_final_state_vecN16_f32_kernel<<<grid, block, shmem>>>(
        sptr, aptr, optr,
        (int)B64, (int)C164, (int)H64, (int)P64
    );
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor mamba2_return_final_state_cuda(torch::Tensor states_cat, torch::Tensor a_last_pad);
"""

custom_ops_lib = load_inline(
    name="custom_mamba2_return_final_state_ext_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mamba2_return_final_state_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        self.custom_ops_lib = custom_ops_lib

    def segsum(self, x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        L = torch.exp(self.segsum(A_blocks))
        _Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blocks, B_blocks, L, X_blocks
        )

        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)  # [B,C,H,P,N]

        if initial_states is None:
            initial_states_ = torch.zeros_like(states[:, :1])  # [B,1,H,P,N]
        else:
            initial_states_ = initial_states
            if initial_states_.dim() == 4:
                initial_states_ = initial_states_.unsqueeze(1)  # [B,1,H,P,N]

        states_cat = torch.cat([initial_states_, states], dim=1)  # [B,C1,H,P,N]

        a_last = A_cumsum[..., -1]          # [B,H,C]
        a_last_pad = F.pad(a_last, (1, 0))  # [B,H,C1]

        if (
            states_cat.is_cuda and a_last_pad.is_cuda
            and states_cat.dtype == torch.float32 and a_last_pad.dtype == torch.float32
            and states_cat.dim() == 5 and a_last_pad.dim() == 3
        ):
            states_cat_c = states_cat.contiguous()
            a_last_pad_c = a_last_pad.contiguous()
            if (
                states_cat_c.is_contiguous() and a_last_pad_c.is_contiguous()
                and a_last_pad_c.size(0) == states_cat_c.size(0)
                and a_last_pad_c.size(1) == states_cat_c.size(2)
                and a_last_pad_c.size(2) == states_cat_c.size(1)
                and states_cat_c.size(4) == 16
            ):
                return self.custom_ops_lib.mamba2_return_final_state_cuda(states_cat_c, a_last_pad_c)

        decay_chunk = torch.exp(self.segsum(a_last_pad))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
        return new_states[:, -1]