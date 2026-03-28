import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# Custom CUDA extension: warp-reduced hidden + output per timestep
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float fast_tanhf(float x) {
    return tanhf(x); // --use_fast_math
}

// Grid: (B, ceil_div(H, WARPS_PER_BLOCK))
// Each warp computes one hidden neuron j for one batch b at one timestep.
// h_out[b,j] = tanh( bi2h[j] + dot(x, Wi2h[j, :I]) + dot(h_in, Wi2h[j, I:]) )
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
void rnn_hidden_warp_kernel(
    const float* __restrict__ x_tb,     // [B, I] for a fixed t
    const float* __restrict__ h_in,     // [B, H]
    const float* __restrict__ Wi2h,     // [H, I+H]
    const float* __restrict__ bi2h,     // [H] or nullptr
    float* __restrict__ h_out,          // [B, H]
    int B, int I, int H
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int j = (int)blockIdx.y * WARPS_PER_BLOCK + warp;
    if (j >= H) return;

    const float* x_row = x_tb + (size_t)b * (size_t)I;
    const float* h_row = h_in + (size_t)b * (size_t)H;
    const float* w_row = Wi2h + (size_t)j * (size_t)(I + H);

    float acc = 0.0f;

    // x part
    for (int k = lane; k < I; k += 32) {
        acc = fmaf(x_row[k], ldg_f32(w_row + k), acc);
    }
    // h part
    const float* w_h = w_row + I;
    for (int k = lane; k < H; k += 32) {
        acc = fmaf(h_row[k], ldg_f32(w_h + k), acc);
    }

    acc = warp_sum(acc);

    if (lane == 0) {
        float bval = bi2h ? ldg_f32(bi2h + j) : 0.0f;
        h_out[(size_t)b * (size_t)H + (size_t)j] = fast_tanhf(acc + bval);
    }
}

// Grid: (B, ceil_div(O, WARPS_PER_BLOCK))
// Each warp computes one output neuron o for one batch b:
// out[b,o] = bh2o[o] + dot(h, Wh2o[o,:])
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
void rnn_out_warp_kernel(
    const float* __restrict__ h,        // [B, H]
    const float* __restrict__ Wh2o,     // [O, H]
    const float* __restrict__ bh2o,     // [O] or nullptr
    float* __restrict__ out_tb,         // [B, O] for a fixed t
    int B, int H, int O
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int o = (int)blockIdx.y * WARPS_PER_BLOCK + warp;
    if (o >= O) return;

    const float* h_row = h + (size_t)b * (size_t)H;
    const float* w_row = Wh2o + (size_t)o * (size_t)H;

    float acc = 0.0f;

    // small unroll by 2 for better ILP
    for (int k = lane; k < H; k += 32) {
        acc = fmaf(h_row[k], ldg_f32(w_row + k), acc);
    }

    acc = warp_sum(acc);

    if (lane == 0) {
        float bval = bh2o ? ldg_f32(bh2o + o) : 0.0f;
        out_tb[(size_t)b * (size_t)O + (size_t)o] = acc + bval;
    }
}

torch::Tensor vanilla_rnn_hidden_cuda(
    torch::Tensor x,    // [T,B,I]
    torch::Tensor h0,   // [B,H]
    torch::Tensor Wi2h, // [H,I+H]
    torch::Tensor bi2h,
    torch::Tensor Wh2o, // [O,H]
    torch::Tensor bh2o
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(h0.is_cuda(), "h0 must be CUDA");
    TORCH_CHECK(Wi2h.is_cuda(), "Wi2h must be CUDA");
    TORCH_CHECK(Wh2o.is_cuda(), "Wh2o must be CUDA");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(h0.dtype() == torch::kFloat32, "h0 must be float32");
    TORCH_CHECK(Wi2h.dtype() == torch::kFloat32, "Wi2h must be float32");
    TORCH_CHECK(Wh2o.dtype() == torch::kFloat32, "Wh2o must be float32");
    TORCH_CHECK(!bi2h.defined() || bi2h.dtype() == torch::kFloat32, "bi2h must be float32");
    TORCH_CHECK(!bh2o.defined() || bh2o.dtype() == torch::kFloat32, "bh2o must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(h0.is_contiguous(), "h0 must be contiguous");
    TORCH_CHECK(Wi2h.is_contiguous(), "Wi2h must be contiguous");
    TORCH_CHECK(Wh2o.is_contiguous(), "Wh2o must be contiguous");
    if (bi2h.defined()) TORCH_CHECK(bi2h.is_contiguous(), "bi2h must be contiguous");
    if (bh2o.defined()) TORCH_CHECK(bh2o.is_contiguous(), "bh2o must be contiguous");

    TORCH_CHECK(x.dim() == 3, "x must be [T,B,I]");
    TORCH_CHECK(h0.dim() == 2, "h0 must be [B,H]");
    TORCH_CHECK(Wi2h.dim() == 2, "Wi2h must be [H,I+H]");
    TORCH_CHECK(Wh2o.dim() == 2, "Wh2o must be [O,H]");

    int64_t T64 = x.size(0), B64 = x.size(1), I64 = x.size(2);
    TORCH_CHECK(h0.size(0) == B64, "batch mismatch between x and h0");
    int64_t H64 = h0.size(1);
    TORCH_CHECK(Wi2h.size(0) == H64, "Wi2h first dim must be hidden_size");
    TORCH_CHECK(Wi2h.size(1) == I64 + H64, "Wi2h second dim must be input_size + hidden_size");
    TORCH_CHECK(!bi2h.defined() || bi2h.numel() == H64, "bi2h must have hidden_size elements");

    int64_t O64 = Wh2o.size(0);
    TORCH_CHECK(Wh2o.size(1) == H64, "Wh2o second dim must be hidden_size");
    TORCH_CHECK(!bh2o.defined() || bh2o.numel() == O64, "bh2o must have output_size elements");

    int T = (int)T64, B = (int)B64, I = (int)I64, H = (int)H64, O = (int)O64;

    auto out = torch::empty({T64, B64, O64}, x.options());

    // ping-pong hidden buffers
    auto h_a = torch::empty({B64, H64}, x.options());
    auto h_b = torch::empty({B64, H64}, x.options());
    h_a.copy_(h0);

    const float* bi_ptr = bi2h.defined() ? bi2h.data_ptr<float>() : nullptr;
    const float* bo_ptr = bh2o.defined() ? bh2o.data_ptr<float>() : nullptr;

    constexpr int WARPS = 4; // 128 threads; tends to reduce regs and improve occupancy
    dim3 block(WARPS * 32);

    dim3 grid_h((unsigned)B, (unsigned)((H + WARPS - 1) / WARPS));
    dim3 grid_o((unsigned)B, (unsigned)((O + WARPS - 1) / WARPS));

    const float* h_in = h_a.data_ptr<float>();
    float* h_out = h_b.data_ptr<float>();

    for (int t = 0; t < T; t++) {
        const float* x_tb = x.data_ptr<float>() + ((size_t)t * (size_t)B) * (size_t)I;
        float* out_tb = out.data_ptr<float>() + ((size_t)t * (size_t)B) * (size_t)O;

        rnn_hidden_warp_kernel<WARPS><<<grid_h, block>>>(
            x_tb, h_in, Wi2h.data_ptr<float>(), bi_ptr, h_out, B, I, H
        );

        rnn_out_warp_kernel<WARPS><<<grid_o, block>>>(
            h_out, Wh2o.data_ptr<float>(), bo_ptr, out_tb, B, H, O
        );

        // swap
        const float* tmp_in = h_in;
        h_in = h_out;
        h_out = (float*)tmp_in; // safe: both are float* originally
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor vanilla_rnn_hidden_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor Wi2h,
    torch::Tensor bi2h,
    torch::Tensor Wh2o,
    torch::Tensor bh2o
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vanilla_rnn_hidden_cuda", &vanilla_rnn_hidden_cuda, "vanilla_rnn_hidden_cuda (warp-reduced, CUDA)");
}
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_vanilla_rnn_hidden_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ----------------------------
# Model using custom op
# ----------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self._ops = custom_ops_lib

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            x = x.cuda()
        if h0.device.type != "cuda":
            h0 = h0.cuda()

        if x.dtype != torch.float32:
            x = x.float()
        if h0.dtype != torch.float32:
            h0 = h0.float()

        x_c = x.contiguous()
        h0_c = h0.contiguous()

        Wi2h = self.i2h.weight.contiguous()
        bi2h = self.i2h.bias.contiguous() if self.i2h.bias is not None else torch.Tensor()
        Wh2o = self.h2o.weight.contiguous()
        bh2o = self.h2o.bias.contiguous() if self.h2o.bias is not None else torch.Tensor()

        return self._ops.vanilla_rnn_hidden_cuda(x_c, h0_c, Wi2h, bi2h, Wh2o, bh2o)