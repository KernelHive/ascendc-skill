import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------
# Helpers (kept in PyTorch)
# -----------------------

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2
    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = x.reshape(b, l * (m + 1))
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2
    logits = torch.einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = logits.reshape(b * h, w, -1)
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5
        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size
        q = q.reshape(q.shape[0], block, block, q.shape[-1])
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.reshape(q.shape[0], block, block, rel_logits_w.shape[-2], rel_logits_w.shape[-1])
        rel_logits_w = rel_logits_w.reshape(q.shape[0], block * block, rel_logits_w.shape[-2] * rel_logits_w.shape[-1])

        q = q.permute(0, 2, 1, 3)
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.reshape(q.shape[0], block, block, rel_logits_h.shape[-2], rel_logits_h.shape[-1])
        rel_logits_h = rel_logits_h.reshape(q.shape[0], block * block, rel_logits_h.shape[-1] * rel_logits_h.shape[-2])
        return rel_logits_w + rel_logits_h

# -----------------------
# Custom CUDA extension
# -----------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be bool")
#define CHECK_INPUT_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)
#define CHECK_INPUT_BOOL(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BOOL(x)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float shared[32]; // up to 1024 threads
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    return v;
}

// Fused halo attention forward:
// sim_ij = dot(q_i, k_j); if mask[j] == True => sim_ij = max_neg; attn = softmax(sim); out_i = sum_j attn_ij * v_j
// Shapes:
//   Q: [B, I, D], K: [B, J, D], V: [B, J, D], Mask: [B, 1, J] (bool, True means masked)
//   Out: [B, I, D]
__global__ void halo_attn_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const bool*  __restrict__ Mask,
    float* __restrict__ Out,
    int B, int I, int J, int D,
    float max_neg
) {
    int b = (int)blockIdx.x;
    int i = (int)blockIdx.y;
    if (b >= B || i >= I) return;

    const float* q = Q + ((b * I + i) * D);
    const bool*  m = Mask + (b * J); // since middle dim is 1
    const float* k_base = K + (b * J * D);
    const float* v_base = V + (b * J * D);
    float* out = Out + ((b * I + i) * D);

    // 1) find max over j of masked sim
    float tmax = -INFINITY;
    for (int j = threadIdx.x; j < J; j += blockDim.x) {
        float s = max_neg;
        if (!m[j]) {
            const float* k = k_base + j * D;
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) dot = fmaf(q[d], k[d], dot);
            s = dot;
        }
        tmax = fmaxf(tmax, s);
    }
    float row_max = block_reduce_max(tmax);
    __shared__ float sh_max;
    if (threadIdx.x == 0) sh_max = row_max;
    __syncthreads();

    // 2) sum exp
    float tsum = 0.0f;
    for (int j = threadIdx.x; j < J; j += blockDim.x) {
        float s = max_neg;
        if (!m[j]) {
            const float* k = k_base + j * D;
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) dot = fmaf(q[d], k[d], dot);
            s = dot;
        }
        // exp(max_neg - row_max) underflows to 0, matching masked_fill then softmax
        tsum += expf(s - sh_max);
    }
    float row_sum = block_reduce_sum(tsum);
    __shared__ float sh_invsum;
    if (threadIdx.x == 0) sh_invsum = 1.0f / row_sum;
    __syncthreads();

    // 3) out accumulation (parallelize over d)
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < J; ++j) {
            float s = max_neg;
            if (!m[j]) {
                const float* k = k_base + j * D;
                float dot = 0.0f;
                #pragma unroll 4
                for (int dd = 0; dd < D; ++dd) dot = fmaf(q[dd], k[dd], dot);
                s = dot;
            }
            float w = expf(s - sh_max) * sh_invsum;
            acc = fmaf(w, v_base[j * D + d], acc);
        }
        out[d] = acc;
    }
}

torch::Tensor halo_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Mask) {
    CHECK_INPUT_FLOAT(Q);
    CHECK_INPUT_FLOAT(K);
    CHECK_INPUT_FLOAT(V);
    CHECK_INPUT_BOOL(Mask);

    TORCH_CHECK(Q.dim() == 3, "Q must be [B, I, D]");
    TORCH_CHECK(K.dim() == 3, "K must be [B, J, D]");
    TORCH_CHECK(V.dim() == 3, "V must be [B, J, D]");
    TORCH_CHECK(Mask.dim() == 3, "Mask must be [B, 1, J]");

    int B = (int)Q.size(0);
    int I = (int)Q.size(1);
    int D = (int)Q.size(2);
    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "K,V batch must match Q");
    TORCH_CHECK(K.size(2) == D && V.size(2) == D, "K,V D must match Q");
    int J = (int)K.size(1);
    TORCH_CHECK(V.size(1) == J, "V J must match K");
    TORCH_CHECK(Mask.size(0) == B && Mask.size(1) == 1 && Mask.size(2) == J, "Mask must be [B,1,J]");

    auto Out = torch::empty({B, I, D}, Q.options());

    // -torch.finfo(float32).max
    float max_neg = -3.402823466e+38f;

    dim3 grid((unsigned)B, (unsigned)I, 1);
    int threads = 256;
    halo_attn_fwd<<<grid, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Mask.data_ptr<bool>(),
        Out.data_ptr<float>(),
        B, I, J, D,
        max_neg
    );

    return Out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor halo_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Mask);
"""

custom_ops_lib = load_inline(
    name="custom_halo_attention_ops_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["halo_attention_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
)

# -----------------------
# Model using custom op
# -----------------------

class ModelNew(nn.Module):
    """
    Same as reference Model, but replaces:
      sim = einsum(q,k) + masked_fill + softmax + einsum(attn,v)
    with a fused CUDA kernel.
    Note: reference code instantiates RelPosEmb but does not use it in forward; we preserve that behavior.
    """
    def __init__(self, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size=block_size,
            rel_size=block_size + (halo_size * 2),
            dim_head=dim_head
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        b, c, h, w = x.shape
        block = self.block_size
        halo = self.halo_size
        heads = self.heads
        device = x.device

        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # queries per block
        q_inp = (
            x.reshape(b, c, h // block, block, w // block, block)
             .permute(0, 2, 4, 3, 5, 1)
             .reshape(b * (h // block) * (w // block), block * block, c)
        )

        # keys/values neighborhoods
        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = kv_inp.reshape(b, c, -1, kv_inp.shape[-1]).permute(0, 3, 2, 1).reshape(b * kv_inp.shape[-1], -1, c)

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # split heads -> [Bwin*heads, I/J, Dh]
        dh = q.shape[-1] // heads
        q = q.reshape(q.shape[0], q.shape[1], heads, dh).permute(0, 2, 1, 3).reshape(q.shape[0] * heads, q.shape[1], dh)
        k = k.reshape(k.shape[0], k.shape[1], heads, dh).permute(0, 2, 1, 3).reshape(k.shape[0] * heads, k.shape[1], dh)
        v = v.reshape(v.shape[0], v.shape[1], heads, dh).permute(0, 2, 1, 3).reshape(v.shape[0] * heads, v.shape[1], dh)

        q = (q * self.scale).contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # mask: True means masked (matches sim.masked_fill_(mask, max_neg))
        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + (halo * 2), stride=block, padding=halo)
        mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
        num_windows = mask.shape[-1]
        mask = (
            mask.permute(0, 3, 1, 2)
                .reshape(b * num_windows, 1, -1)
                .expand(-1, heads, -1)
                .reshape(b * num_windows * heads, 1, -1)
                .bool()
                .contiguous()
        )

        out = self.custom_ops_lib.halo_attention_forward_cuda(q, k, v, mask)

        # merge heads
        out = out.reshape(-1, heads, out.shape[1], out.shape[2]).permute(0, 2, 1, 3).reshape(-1, out.shape[1], heads * out.shape[2])
        out = self.to_out(out)

        # merge blocks back
        out = out.reshape(b, h // block, w // block, block, block, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        return out