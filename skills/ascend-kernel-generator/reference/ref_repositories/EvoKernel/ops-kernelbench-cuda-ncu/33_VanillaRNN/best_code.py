import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA extension: streaming warp-GEMV + optional fusion
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float fast_tanhf(float x) {
    // compiled with --use_fast_math
    return tanhf(x);
}

__device__ __forceinline__ float ldg_f32(const float* p) {
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

// -------------------------------
// Hidden update: warp streaming GEMV
// Mapping:
//  grid.x = B
//  grid.y = ceil_div(H, WARPS_PER_BLOCK * OUTS_PER_WARP)
//  block = WARPS_PER_BLOCK * 32 threads
// Each warp computes OUTS_PER_WARP hidden outputs.
// -------------------------------
template<int WARPS_PER_BLOCK, int OUTS_PER_WARP, int UNROLL_K4>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void rnn_hidden_stream_kernel(
    const float* __restrict__ x,     // [B,I]
    const float* __restrict__ h,     // [B,H]
    const float* __restrict__ Wi2h,  // [H, I+H]
    const float* __restrict__ bi2h,  // [H] or nullptr
    float* __restrict__ h_new,       // [B,H]
    int B, int I, int H
) {
    const int b = (int)blockIdx.x;
    if (b >= B) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    constexpr int OUTS_PER_BLOCK = WARPS_PER_BLOCK * OUTS_PER_WARP;
    const int out_base = (int)blockIdx.y * OUTS_PER_BLOCK;
    const int warp_out_base = out_base + warp * OUTS_PER_WARP;

    const float* x_row = x + (size_t)b * (size_t)I;
    const float* h_row = h + (size_t)b * (size_t)H;

    // accumulators
    float acc[OUTS_PER_WARP];
#pragma unroll
    for (int i = 0; i < OUTS_PER_WARP; i++) acc[i] = 0.0f;

    // x part: k in [0, I)
    // each lane iterates over k = lane, lane+32, ...
    for (int k = lane; k < I; k += 32) {
        float xv = x_row[k];

        // unroll over multiple outputs; weights contiguous per output row
#pragma unroll
        for (int oi = 0; oi < OUTS_PER_WARP; oi++) {
            int out = warp_out_base + oi;
            if (out < H) {
                const float* w_row = Wi2h + (size_t)out * (size_t)(I + H);
                acc[oi] = fmaf(xv, ldg_f32(w_row + k), acc[oi]);
            }
        }
    }

    // h part: weights offset by I, k in [0, H)
    for (int k = lane; k < H; k += 32) {
        float hv = h_row[k];
#pragma unroll
        for (int oi = 0; oi < OUTS_PER_WARP; oi++) {
            int out = warp_out_base + oi;
            if (out < H) {
                const float* w_row = Wi2h + (size_t)out * (size_t)(I + H) + (size_t)I;
                acc[oi] = fmaf(hv, ldg_f32(w_row + k), acc[oi]);
            }
        }
    }

    // warp reduce
#pragma unroll
    for (int oi = 0; oi < OUTS_PER_WARP; oi++) acc[oi] = warp_reduce_sum(acc[oi]);

    if (lane == 0) {
#pragma unroll
        for (int oi = 0; oi < OUTS_PER_WARP; oi++) {
            int out = warp_out_base + oi;
            if (out < H) {
                float b0 = bi2h ? ldg_f32(bi2h + out) : 0.0f;
                h_new[(size_t)b * (size_t)H + (size_t)out] = fast_tanhf(acc[oi] + b0);
            }
        }
    }
}

// -------------------------------
// Output projection: warp streaming GEMV, 2 outputs/warp
// Mapping:
//  grid.x = B
//  grid.y = ceil_div(O, WARPS_PER_BLOCK * OUTS_PER_WARP_O)
// -------------------------------
template<int WARPS_PER_BLOCK, int OUTS_PER_WARP_O>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void rnn_out_stream_kernel(
    const float* __restrict__ h_new, // [B,H]
    const float* __restrict__ Wh2o,  // [O,H]
    const float* __restrict__ bh2o,  // [O] or nullptr
    float* __restrict__ out,         // [B,O]
    int B, int H, int O
) {
    const int b = (int)blockIdx.x;
    if (b >= B) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    constexpr int OUTS_PER_BLOCK = WARPS_PER_BLOCK * OUTS_PER_WARP_O;
    const int o_base = (int)blockIdx.y * OUTS_PER_BLOCK;
    const int warp_o_base = o_base + warp * OUTS_PER_WARP_O;

    const float* hn_row = h_new + (size_t)b * (size_t)H;

    float acc[OUTS_PER_WARP_O];
#pragma unroll
    for (int i = 0; i < OUTS_PER_WARP_O; i++) acc[i] = 0.0f;

    // attempt float4 vectorization over H when aligned
    const int H4 = H & ~3;
    const bool aligned = ((((uintptr_t)hn_row) & 0xF) == 0);

    // process float4 chunks; each lane handles base = lane*4 + n*128
    for (int base = lane * 4; base < H4; base += 32 * 4) {
        float4 hv4;
        if (aligned) {
            hv4 = *reinterpret_cast<const float4*>(hn_row + base);
        } else {
            hv4.x = hn_row[base + 0];
            hv4.y = hn_row[base + 1];
            hv4.z = hn_row[base + 2];
            hv4.w = hn_row[base + 3];
        }

#pragma unroll
        for (int oi = 0; oi < OUTS_PER_WARP_O; oi++) {
            int oidx = warp_o_base + oi;
            if (oidx < O) {
                const float* w_row = Wh2o + (size_t)oidx * (size_t)H;
                float4 wv4;
                if ((((uintptr_t)(w_row + base)) & 0xF) == 0) {
                    wv4 = *reinterpret_cast<const float4*>(w_row + base);
                } else {
                    wv4.x = ldg_f32(w_row + base + 0);
                    wv4.y = ldg_f32(w_row + base + 1);
                    wv4.z = ldg_f32(w_row + base + 2);
                    wv4.w = ldg_f32(w_row + base + 3);
                }
                acc[oi] = fmaf(hv4.x, wv4.x, acc[oi]);
                acc[oi] = fmaf(hv4.y, wv4.y, acc[oi]);
                acc[oi] = fmaf(hv4.z, wv4.z, acc[oi]);
                acc[oi] = fmaf(hv4.w, wv4.w, acc[oi]);
            }
        }
    }

    // tail
    for (int k = H4 + lane; k < H; k += 32) {
        float hv = hn_row[k];
#pragma unroll
        for (int oi = 0; oi < OUTS_PER_WARP_O; oi++) {
            int oidx = warp_o_base + oi;
            if (oidx < O) {
                const float* w_row = Wh2o + (size_t)oidx * (size_t)H;
                acc[oi] = fmaf(hv, ldg_f32(w_row + k), acc[oi]);
            }
        }
    }

#pragma unroll
    for (int oi = 0; oi < OUTS_PER_WARP_O; oi++) acc[oi] = warp_reduce_sum(acc[oi]);

    if (lane == 0) {
#pragma unroll
        for (int oi = 0; oi < OUTS_PER_WARP_O; oi++) {
            int oidx = warp_o_base + oi;
            if (oidx < O) {
                float b0 = bh2o ? ldg_f32(bh2o + oidx) : 0.0f;
                out[(size_t)b * (size_t)O + (size_t)oidx] = acc[oi] + b0;
            }
        }
    }
}

std::vector<torch::Tensor> vanilla_rnn_opt_cuda(
    torch::Tensor x,        // [B,I]
    torch::Tensor h,        // [B,H]
    torch::Tensor Wi2h,     // [H, I+H]
    torch::Tensor bi2h,     // [H]
    torch::Tensor Wh2o,     // [O, H]
    torch::Tensor bh2o      // [O]
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(h.is_cuda(), "h must be CUDA");
    TORCH_CHECK(Wi2h.is_cuda(), "Wi2h must be CUDA");
    TORCH_CHECK(Wh2o.is_cuda(), "Wh2o must be CUDA");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(h.dtype() == torch::kFloat32, "h must be float32");
    TORCH_CHECK(Wi2h.dtype() == torch::kFloat32, "Wi2h must be float32");
    TORCH_CHECK(Wh2o.dtype() == torch::kFloat32, "Wh2o must be float32");
    TORCH_CHECK(!bi2h.defined() || bi2h.dtype() == torch::kFloat32, "bi2h must be float32");
    TORCH_CHECK(!bh2o.defined() || bh2o.dtype() == torch::kFloat32, "bh2o must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(h.is_contiguous(), "h must be contiguous");
    TORCH_CHECK(Wi2h.is_contiguous(), "Wi2h must be contiguous");
    TORCH_CHECK(Wh2o.is_contiguous(), "Wh2o must be contiguous");
    if (bi2h.defined()) TORCH_CHECK(bi2h.is_contiguous(), "bi2h must be contiguous");
    if (bh2o.defined()) TORCH_CHECK(bh2o.is_contiguous(), "bh2o must be contiguous");

    int64_t B64 = x.size(0);
    int64_t I64 = x.size(1);
    int64_t H64 = h.size(1);
    TORCH_CHECK(h.size(0) == B64, "batch mismatch between x and h");
    TORCH_CHECK(Wi2h.size(0) == H64, "Wi2h first dim must be hidden_size");
    TORCH_CHECK(Wi2h.size(1) == I64 + H64, "Wi2h second dim must be input_size + hidden_size");
    TORCH_CHECK(!bi2h.defined() || (bi2h.numel() == H64), "bi2h must have size hidden_size");
    int64_t O64 = Wh2o.size(0);
    TORCH_CHECK(Wh2o.size(1) == H64, "Wh2o second dim must be hidden_size");
    TORCH_CHECK(!bh2o.defined() || (bh2o.numel() == O64), "bh2o must have size output_size");

    int B = (int)B64;
    int I = (int)I64;
    int H = (int)H64;
    int O = (int)O64;

    auto h_new = torch::empty({B64, H64}, x.options());
    auto out   = torch::empty({B64, O64}, x.options());

    const float* bi_ptr = bi2h.defined() ? bi2h.data_ptr<float>() : nullptr;
    const float* bo_ptr = bh2o.defined() ? bh2o.data_ptr<float>() : nullptr;

    // Tune: 256 threads, each warp does 4 hidden outs, and 2 output outs.
    constexpr int WARPS = 8;          // 256 threads
    constexpr int OUTS_H = 4;         // hidden outputs per warp
    constexpr int OUTS_O = 2;         // output outputs per warp

    dim3 block(WARPS * 32);
    dim3 grid_h((unsigned)B, (unsigned)((H + (WARPS*OUTS_H) - 1) / (WARPS*OUTS_H)));
    dim3 grid_o((unsigned)B, (unsigned)((O + (WARPS*OUTS_O) - 1) / (WARPS*OUTS_O)));

    rnn_hidden_stream_kernel<WARPS, OUTS_H, 0><<<grid_h, block>>>(
        x.data_ptr<float>(),
        h.data_ptr<float>(),
        Wi2h.data_ptr<float>(),
        bi_ptr,
        h_new.data_ptr<float>(),
        B, I, H
    );

    rnn_out_stream_kernel<WARPS, OUTS_O><<<grid_o, block>>>(
        h_new.data_ptr<float>(),
        Wh2o.data_ptr<float>(),
        bo_ptr,
        out.data_ptr<float>(),
        B, H, O
    );

    return {out, h_new};
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> vanilla_rnn_opt_cuda(
    torch::Tensor x,
    torch::Tensor h,
    torch::Tensor Wi2h,
    torch::Tensor bi2h,
    torch::Tensor Wh2o,
    torch::Tensor bh2o
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vanilla_rnn_opt_cuda", &vanilla_rnn_opt_cuda, "Optimized Vanilla RNN step (CUDA, streaming warp-GEMV)");
}
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ------------------------------------------------------------
# Model using optimized custom op
# ------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h_weight = nn.Parameter(torch.empty(hidden_size, input_size + hidden_size))
        self.i2h_bias = nn.Parameter(torch.empty(hidden_size))

        self.h2o_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.h2o_bias = nn.Parameter(torch.empty(output_size))

        self.register_buffer("hidden", torch.randn(batch_size, hidden_size))

        nn.init.kaiming_uniform_(self.i2h_weight, a=5 ** 0.5)
        nn.init.uniform_(self.i2h_bias, -0.01, 0.01)
        nn.init.kaiming_uniform_(self.h2o_weight, a=5 ** 0.5)
        nn.init.uniform_(self.h2o_bias, -0.01, 0.01)

        self._ops = custom_ops_lib

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)

        if self.hidden.device != x.device:
            self.hidden = self.hidden.to(device=x.device)

        if x.dtype != torch.float32:
            x = x.float()
        if self.hidden.dtype != torch.float32:
            self.hidden = self.hidden.float()

        out, h_new = self._ops.vanilla_rnn_opt_cuda(
            x.contiguous(),
            self.hidden.contiguous(),
            self.i2h_weight.contiguous(),
            self.i2h_bias.contiguous(),
            self.h2o_weight.contiguous(),
            self.h2o_bias.contiguous(),
        )
        self.hidden = h_new
        return out