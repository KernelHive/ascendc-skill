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

#include <ATen/cuda/CUDAContext.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static inline bool is_aligned_ptr(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

__device__ __forceinline__ float4 load_f4_aligned(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void store_f4_aligned(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

// y_diag/y_off: [B,C,L,H,P] contiguous
// y: [B,T,H,P] contiguous with T=C*L (matches flatten of [B,C,L,H,P] where (c,l) -> t)
__global__ __launch_bounds__(256, 3) void mamba2_return_y_vec4_2d_kernel(
    const float* __restrict__ y_diag,
    const float* __restrict__ y_off,
    float* __restrict__ y,
    int outer, // B*T*H
    int P      // tail
) {
    int p4 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; // in float4 units
    int P4 = P >> 2;
    if (p4 >= P4) return;

    int o = (int)blockIdx.y;
    if (o >= outer) return;

    // base in scalar floats
    int base = o * P + (p4 << 2);

    float4 a = load_f4_aligned(y_diag + base);
    float4 b = load_f4_aligned(y_off  + base);
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    store_f4_aligned(y + base, a);
}

__global__ __launch_bounds__(256, 3) void mamba2_return_y_scalar_2d_kernel(
    const float* __restrict__ y_diag,
    const float* __restrict__ y_off,
    float* __restrict__ y,
    int outer,
    int P
) {
    int p = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int o = (int)blockIdx.y;
    if (p >= P || o >= outer) return;
    int idx = o * P + p;
    y[idx] = y_diag[idx] + y_off[idx];
}

torch::Tensor mamba2_return_y_cuda(torch::Tensor y_diag, torch::Tensor y_off) {
    CHECK_CUDA(y_diag);
    CHECK_CUDA(y_off);
    CHECK_CONTIGUOUS(y_diag);
    CHECK_CONTIGUOUS(y_off);
    CHECK_FLOAT(y_diag);
    CHECK_FLOAT(y_off);

    TORCH_CHECK(y_diag.sizes() == y_off.sizes(), "y_diag and y_off must have same shape");
    TORCH_CHECK(y_diag.dim() == 5, "Expected y_diag/y_off to be 5D: [B, C, L, H, P]");

    int64_t B64 = y_diag.size(0);
    int64_t C64 = y_diag.size(1);
    int64_t L64 = y_diag.size(2);
    int64_t H64 = y_diag.size(3);
    int64_t P64 = y_diag.size(4);

    int64_t T64 = C64 * L64;
    auto y = torch::empty({B64, T64, H64, P64}, y_diag.options());
    if (B64 == 0 || C64 == 0 || L64 == 0 || H64 == 0 || P64 == 0) return y;

    // Flatten [B,C,L,H,P] -> [outer, P], where outer = B*T*H
    // This is safe because [C,L] are contiguous and only reinterpreted in output shape.
    int outer = (int)(B64 * T64 * H64);
    int P = (int)P64;

    const float* dptr = (const float*)y_diag.data_ptr<float>();
    const float* optr = (const float*)y_off.data_ptr<float>();
    float* yptr = (float*)y.data_ptr<float>();

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bool vec4_ok = ((P & 3) == 0) && is_aligned_ptr(dptr, 16) && is_aligned_ptr(optr, 16) && is_aligned_ptr(yptr, 16);

    if (vec4_ok) {
        int P4 = P >> 2;
        int threads = 256;
        int blocks_x = (P4 + threads - 1) / threads;
        // grid.y spans outer; this is the main source of parallelism
        dim3 grid((unsigned)blocks_x, (unsigned)outer, 1);
        dim3 block((unsigned)threads, 1, 1);
        mamba2_return_y_vec4_2d_kernel<<<grid, block, 0, stream>>>(dptr, optr, yptr, outer, P);
    } else {
        int threads = 256;
        int blocks_x = (P + threads - 1) / threads;
        dim3 grid((unsigned)blocks_x, (unsigned)outer, 1);
        dim3 block((unsigned)threads, 1, 1);
        mamba2_return_y_scalar_2d_kernel<<<grid, block, 0, stream>>>(dptr, optr, yptr, outer, P);
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor mamba2_return_y_cuda(torch::Tensor y_diag, torch::Tensor y_off);
"""

custom_ops_lib = load_inline(
    name="custom_mamba2_return_y_ext_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mamba2_return_y_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
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
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blocks, B_blocks, L, X_blocks
        )

        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)

        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C_blocks, states, state_decay_out)

        if (
            Y_diag.is_cuda and Y_off.is_cuda
            and Y_diag.dtype == torch.float32 and Y_off.dtype == torch.float32
            and Y_diag.is_contiguous() and Y_off.is_contiguous()
            and Y_diag.dim() == 5 and Y_off.dim() == 5
            and Y_diag.shape == Y_off.shape
        ):
            return self.custom_ops_lib.mamba2_return_y_cuda(Y_diag, Y_off)

        return rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")