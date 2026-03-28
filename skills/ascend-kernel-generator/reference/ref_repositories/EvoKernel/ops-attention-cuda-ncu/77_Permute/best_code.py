import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA op: moe_permute_bf16_cuda
#
# Key changes vs baseline:
# 1) Replace occupancy-starved per-expert serial scan with global stable sort by expert:
#    - Build (key=expert_id, val=slot_index) for all slots.
#    - CUB stable radix sort by expert_id -> per-expert contiguous segments in stable slot order.
# 2) Compute counts from sorted keys, then padded offsets.
# 3) Build sorted_ids (slot for each output row, with padding by repeating last slot) in parallel.
# 4) Fused m_indices + gather, with bf16x8 vectorized loads/stores along K when aligned (16B) and K%8==0.
# 5) inv_perm computed as stable argsort(sorted_ids)[:Nslots] via CUB radix sort on (key=slot, val=position).
# 6) Avoid full stream sync: read back only total_padded using pinned async memcpy + CUDA event.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <vector>
#include <cub/cub.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_I64(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be int64")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)
#define CHECK_INPUT_I32(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_I32(x)

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  #define LDG(x) __ldg(x)
#else
  #define LDG(x) (*(x))
#endif

__global__ void build_keys_vals_kernel(
    const int* __restrict__ flat_ids, // [Nslots]
    int* __restrict__ keys,           // [Nslots]
    int* __restrict__ vals,           // [Nslots]
    int Nslots
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < Nslots) {
        keys[i] = LDG(flat_ids + i);
        vals[i] = i;
    }
}

__global__ void counts_from_keys_kernel(
    const int* __restrict__ keys_sorted, // [Nslots]
    int64_t* __restrict__ counts,        // [G]
    int Nslots, int G
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < Nslots) {
        int g = LDG(keys_sorted + i);
        if ((unsigned)g < (unsigned)G) {
            atomicAdd((unsigned long long*)&counts[g], 1ULL);
        }
    }
}

// Single-thread prefix over small G.
__global__ void compute_padded_offsets_kernel(
    const int64_t* __restrict__ counts,  // [G]
    int64_t* __restrict__ padded,        // [G]
    int64_t* __restrict__ offsets,       // [G+1]
    int64_t* __restrict__ expert_starts, // [G+1] starts within sorted-by-expert slots (un-padded)
    int64_t* __restrict__ total_padded,  // [1]
    int G, int block_m
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int64_t running_out = 0;
        int64_t running_in  = 0;
        for (int g = 0; g < G; ++g) {
            int64_t c = counts[g];
            int64_t p = ((c + (int64_t)block_m - 1) / (int64_t)block_m) * (int64_t)block_m;
            padded[g] = p;
            offsets[g] = running_out;
            expert_starts[g] = running_in;
            running_out += p;
            running_in  += c;
        }
        offsets[G] = running_out;
        expert_starts[G] = running_in;
        total_padded[0] = running_out;
    }
}

// Build sorted_ids for each output row in parallel.
// For row r, find expert g s.t. offsets[g] <= r < offsets[g+1] (linear scan, G small).
// Then i = r - offsets[g]. If i < counts[g], slot = slots_sorted_by_expert[expert_starts[g] + i]
// else slot = last slot in expert segment if counts[g]>0 else 0.
__global__ void build_sorted_ids_from_sorted_by_expert_kernel(
    const int* __restrict__ slots_sorted_by_expert, // [Nslots] (vals from sort by expert)
    const int64_t* __restrict__ counts,             // [G]
    const int64_t* __restrict__ offsets,            // [G+1]
    const int64_t* __restrict__ expert_starts,      // [G+1]
    int64_t* __restrict__ sorted_ids,               // [total_padded]
    int* __restrict__ m_indices,                    // [total_padded]
    int64_t total_padded,
    int Nslots, int G
) {
    int64_t r = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (r >= total_padded) return;

    int g = 0;
    // Linear scan: G is typically 64.
    #pragma unroll 1
    for (int gg = 0; gg < G; ++gg) {
        int64_t b = LDG(offsets + gg);
        int64_t e = LDG(offsets + gg + 1);
        if (r >= b && r < e) { g = gg; break; }
    }

    int64_t base = LDG(offsets + g);
    int64_t i_in = r - base;
    int64_t c = LDG(counts + g);
    int64_t start = LDG(expert_starts + g);

    int64_t slot64 = 0;
    if (c > 0) {
        int idx;
        if (i_in < c) idx = LDG(slots_sorted_by_expert + (int)(start + i_in));
        else          idx = LDG(slots_sorted_by_expert + (int)(start + (c - 1)));
        slot64 = (int64_t)idx;
    } else {
        slot64 = 0;
    }

    // Clamp defensively
    if (slot64 < 0) slot64 = 0;
    if (slot64 >= (int64_t)Nslots) slot64 = (int64_t)(Nslots - 1);

    sorted_ids[r] = slot64;
    m_indices[r] = g;
}

__global__ void gather_bf16_fused_kernel(
    const __nv_bfloat16* __restrict__ hidden, // [M,K]
    const int64_t* __restrict__ sorted_ids,   // [total_padded]
    __nv_bfloat16* __restrict__ out,          // [total_padded,K]
    int64_t total_padded,
    int M, int K, int topk, int Nslots,
    int vec8_ok
) {
    // 2D flatten: row-major over (row, k8) or (row, k)
    if (vec8_ok) {
        int K8 = K >> 3; // 8 bf16 per 16B
        int64_t linear = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        int64_t total = total_padded * (int64_t)K8;
        if (linear >= total) return;

        int64_t row = linear / (int64_t)K8;
        int k8 = (int)(linear - row * (int64_t)K8);

        int64_t slot = LDG(sorted_ids + row);
        if ((uint64_t)slot >= (uint64_t)Nslots) slot = 0;

        int token = (int)(slot / (int64_t)topk);
        if (token < 0) token = 0;
        if (token >= M) token = M - 1;

        const uint4* h4 = reinterpret_cast<const uint4*>(hidden + row*0); // dummy to silence warnings
        const uint4* src4 = reinterpret_cast<const uint4*>(hidden + (int64_t)token * (int64_t)K);
        uint4* dst4 = reinterpret_cast<uint4*>(out + (int64_t)row * (int64_t)K);
        dst4[k8] = src4[k8];
    } else {
        int64_t linear = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        int64_t total = total_padded * (int64_t)K;
        if (linear >= total) return;

        int64_t row = linear / (int64_t)K;
        int k = (int)(linear - row * (int64_t)K);

        int64_t slot = LDG(sorted_ids + row);
        if ((uint64_t)slot >= (uint64_t)Nslots) slot = 0;

        int token = (int)(slot / (int64_t)topk);
        if (token < 0) token = 0;
        if (token >= M) token = M - 1;

        out[row * (int64_t)K + (int64_t)k] =
            hidden[(int64_t)token * (int64_t)K + (int64_t)k];
    }
}

std::vector<torch::Tensor> moe_permute_bf16_cuda(
    torch::Tensor hidden_states, // [M,K] bf16
    torch::Tensor topk_ids,      // [M,topk] int32
    int64_t num_groups,
    int64_t topk,
    int64_t block_m
) {
    CHECK_INPUT_BF16(hidden_states);
    CHECK_INPUT_I32(topk_ids);

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [M,K]");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be [M,topk]");
    TORCH_CHECK(topk_ids.size(0) == hidden_states.size(0), "topk_ids first dim must match M");
    TORCH_CHECK(topk_ids.size(1) == topk, "topk mismatch with topk_ids second dim");
    TORCH_CHECK(num_groups > 0 && topk > 0 && block_m > 0, "num_groups/topk/block_m must be > 0");

    int64_t M64 = hidden_states.size(0);
    int64_t K64 = hidden_states.size(1);
    int64_t Nslots64 = M64 * topk;

    TORCH_CHECK(M64 > 0 && K64 > 0, "invalid sizes");
    TORCH_CHECK(Nslots64 <= (int64_t)INT_MAX, "Nslots too large");
    TORCH_CHECK(num_groups <= (int64_t)INT_MAX, "num_groups too large");
    TORCH_CHECK(M64 <= (int64_t)INT_MAX && K64 <= (int64_t)INT_MAX, "sizes too large");

    int M = (int)M64;
    int K = (int)K64;
    int Nslots = (int)Nslots64;
    int G = (int)num_groups;

    auto device = hidden_states.device();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto opts_i64 = torch::TensorOptions().device(device).dtype(torch::kInt64);
    auto opts_i32 = torch::TensorOptions().device(device).dtype(torch::kInt32);

    auto flat_ids = topk_ids.reshape({Nslots}).contiguous(); // int32 [Nslots]

    // (1) Build key/val
    auto keys_in = torch::empty({Nslots64}, opts_i32);
    auto vals_in = torch::empty({Nslots64}, opts_i32);
    {
        int threads = 256;
        int blocks = div_up_int(Nslots, threads);
        build_keys_vals_kernel<<<blocks, threads, 0, stream>>>(
            (const int*)flat_ids.data_ptr<int>(),
            (int*)keys_in.data_ptr<int>(),
            (int*)vals_in.data_ptr<int>(),
            Nslots
        );
    }

    // (2) Stable sort pairs by expert id
    auto keys_out = torch::empty_like(keys_in);
    auto vals_out = torch::empty_like(vals_in); // slot indices sorted by expert, stable
    size_t temp_bytes_sort1 = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes_sort1,
        (const int*)keys_in.data_ptr<int>(), (int*)keys_out.data_ptr<int>(),
        (const int*)vals_in.data_ptr<int>(), (int*)vals_out.data_ptr<int>(),
        Nslots,
        0, 32,
        stream
    );
    auto temp1 = torch::empty({(int64_t)temp_bytes_sort1},
                              torch::TensorOptions().device(device).dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortPairs(
        (void*)temp1.data_ptr<uint8_t>(), temp_bytes_sort1,
        (const int*)keys_in.data_ptr<int>(), (int*)keys_out.data_ptr<int>(),
        (const int*)vals_in.data_ptr<int>(), (int*)vals_out.data_ptr<int>(),
        Nslots,
        0, 32,
        stream
    );

    // (3) counts from sorted keys
    auto counts = torch::zeros({(int64_t)G}, opts_i64);
    {
        int threads = 256;
        int blocks = div_up_int(Nslots, threads);
        counts_from_keys_kernel<<<blocks, threads, 0, stream>>>(
            (const int*)keys_out.data_ptr<int>(),
            (int64_t*)counts.data_ptr<int64_t>(),
            Nslots, G
        );
    }

    // (4) padded/offsets/expert_starts/total_padded (single thread)
    auto padded  = torch::empty({(int64_t)G}, opts_i64);
    auto offsets = torch::empty({(int64_t)G + 1}, opts_i64);
    auto expert_starts = torch::empty({(int64_t)G + 1}, opts_i64);
    auto total_padded_dev = torch::empty({1}, opts_i64);
    compute_padded_offsets_kernel<<<1, 1, 0, stream>>>(
        (const int64_t*)counts.data_ptr<int64_t>(),
        (int64_t*)padded.data_ptr<int64_t>(),
        (int64_t*)offsets.data_ptr<int64_t>(),
        (int64_t*)expert_starts.data_ptr<int64_t>(),
        (int64_t*)total_padded_dev.data_ptr<int64_t>(),
        G, (int)block_m
    );

    // (5) Read back total_padded with pinned async copy + event (avoid full device sync)
    int64_t* h_total_padded = nullptr;
    AT_CUDA_CHECK(cudaHostAlloc((void**)&h_total_padded, sizeof(int64_t), cudaHostAllocPortable));
    cudaEvent_t evt;
    AT_CUDA_CHECK(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
    AT_CUDA_CHECK(cudaMemcpyAsync(
        h_total_padded,
        (const void*)total_padded_dev.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    AT_CUDA_CHECK(cudaEventRecord(evt, stream));
    AT_CUDA_CHECK(cudaEventSynchronize(evt));
    AT_CUDA_CHECK(cudaEventDestroy(evt));

    int64_t total_padded64 = *h_total_padded;
    AT_CUDA_CHECK(cudaFreeHost(h_total_padded));

    TORCH_CHECK(total_padded64 >= 0, "total_padded must be non-negative");
    TORCH_CHECK(total_padded64 <= (int64_t)INT_MAX, "total_padded too large");
    int64_t total_padded_host = total_padded64;

    // (6) Allocate exact outputs now
    auto sorted_ids = torch::empty({total_padded_host}, opts_i64);
    auto m_indices  = torch::empty({total_padded_host}, opts_i32);
    auto out        = torch::empty({total_padded_host, K64}, hidden_states.options());

    // (7) Build sorted_ids + m_indices in parallel
    {
        int threads = 256;
        int blocks = (int)((total_padded_host + threads - 1) / threads);
        build_sorted_ids_from_sorted_by_expert_kernel<<<blocks, threads, 0, stream>>>(
            (const int*)vals_out.data_ptr<int>(),
            (const int64_t*)counts.data_ptr<int64_t>(),
            (const int64_t*)offsets.data_ptr<int64_t>(),
            (const int64_t*)expert_starts.data_ptr<int64_t>(),
            (int64_t*)sorted_ids.data_ptr<int64_t>(),
            (int*)m_indices.data_ptr<int>(),
            total_padded_host,
            Nslots, G
        );
    }

    // (8) Fused gather (vectorized bf16x8 when possible)
    uintptr_t hptr = (uintptr_t)hidden_states.data_ptr<at::BFloat16>();
    uintptr_t optr = (uintptr_t)out.data_ptr<at::BFloat16>();
    int vec8_ok = ((K % 8) == 0) && ((hptr % 16) == 0) && ((optr % 16) == 0);

    {
        int threads = 256;
        int64_t work = vec8_ok ? (total_padded_host * (K64 / 8)) : (total_padded_host * K64);
        int blocks = (int)((work + threads - 1) / threads);
        gather_bf16_fused_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)hidden_states.data_ptr<at::BFloat16>(),
            (const int64_t*)sorted_ids.data_ptr<int64_t>(),
            (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
            total_padded_host,
            M, K, (int)topk, Nslots,
            vec8_ok
        );
    }

    // (9) inv_perm = argsort(sorted_ids)[:Nslots] via stable sort pairs (key=slot, val=pos)
    auto inv_keys_in  = sorted_ids; // int64
    auto inv_vals_in  = torch::arange(total_padded_host, opts_i64);
    auto inv_keys_out = torch::empty_like(inv_keys_in);
    auto inv_vals_out = torch::empty_like(inv_vals_in);

    size_t temp_bytes_sort2 = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes_sort2,
        (const int64_t*)inv_keys_in.data_ptr<int64_t>(),
        (int64_t*)inv_keys_out.data_ptr<int64_t>(),
        (const int64_t*)inv_vals_in.data_ptr<int64_t>(),
        (int64_t*)inv_vals_out.data_ptr<int64_t>(),
        (int)total_padded_host,
        0, 64,
        stream
    );
    auto temp2 = torch::empty({(int64_t)temp_bytes_sort2},
                              torch::TensorOptions().device(device).dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortPairs(
        (void*)temp2.data_ptr<uint8_t>(), temp_bytes_sort2,
        (const int64_t*)inv_keys_in.data_ptr<int64_t>(),
        (int64_t*)inv_keys_out.data_ptr<int64_t>(),
        (const int64_t*)inv_vals_in.data_ptr<int64_t>(),
        (int64_t*)inv_vals_out.data_ptr<int64_t>(),
        (int)total_padded_host,
        0, 64,
        stream
    );

    auto inv_perm = inv_vals_out.narrow(0, 0, Nslots64).contiguous();

    return {out, m_indices, inv_perm};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> moe_permute_bf16_cuda(torch::Tensor hidden_states, torch::Tensor topk_ids, int64_t num_groups, int64_t topk, int64_t block_m);
"""

custom_ops_lib = load_inline(
    name="custom_moe_permute_ops_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["moe_permute_bf16_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    MoE Permute using optimized custom CUDA implementation.

    Fast path requires:
      hidden_states: CUDA, bfloat16, contiguous, [M,K]
      topk_ids:      CUDA, int32, contiguous, [M,topk]
    Returns:
      output:   [total_padded, K] bfloat16
      m_indices:[total_padded] int32
      inv_perm: [M*topk] int64
    """

    def __init__(self, num_groups: int, topk: int, block_m: int):
        super().__init__()
        self.num_groups = int(num_groups)
        self.topk = int(topk)
        self.block_m = int(block_m)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor):
        if (
            hidden_states.is_cuda
            and topk_ids.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and hidden_states.dim() == 2
            and topk_ids.dim() == 2
            and hidden_states.is_contiguous()
            and topk_ids.is_contiguous()
            and topk_ids.size(0) == hidden_states.size(0)
            and topk_ids.size(1) == self.topk
        ):
            return self.custom_ops_lib.moe_permute_bf16_cuda(
                hidden_states, topk_ids, self.num_groups, self.topk, self.block_m
            )

        # Reference fallback
        M, K = hidden_states.shape
        num_slots = M * self.topk

        flat_ids = topk_ids.reshape(-1)
        counts = torch.zeros(self.num_groups, dtype=torch.int64, device=hidden_states.device)
        for i in range(self.num_groups):
            counts[i] = (flat_ids == i).sum()

        padded_counts = ((counts + self.block_m - 1) // self.block_m) * self.block_m

        sorted_token_ids = []
        m_indices_list = []
        for expert_id in range(self.num_groups):
            mask = (flat_ids == expert_id)
            token_slots = torch.where(mask)[0]
            pad_count = int(padded_counts[expert_id].item()) - int(token_slots.numel())
            if token_slots.numel() > 0:
                padded = (
                    torch.cat([token_slots, token_slots[-1:].expand(pad_count)])
                    if pad_count > 0
                    else token_slots
                )
            else:
                padded = torch.zeros(
                    int(padded_counts[expert_id].item()),
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            sorted_token_ids.append(padded)
            m_indices_list.append(
                torch.full(
                    (int(padded_counts[expert_id].item()),),
                    expert_id,
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            )

        sorted_token_ids = torch.cat(sorted_token_ids)
        m_indices = torch.cat(m_indices_list)

        sorted_token_ids = sorted_token_ids.clamp(max=num_slots - 1)
        inv_perm = torch.argsort(sorted_token_ids)[:num_slots]

        source_token_idx = sorted_token_ids // self.topk
        output = hidden_states[source_token_idx]

        return output, m_indices, inv_perm