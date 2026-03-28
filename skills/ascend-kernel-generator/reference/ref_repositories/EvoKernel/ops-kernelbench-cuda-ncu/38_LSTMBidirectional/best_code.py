import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

// Fused op for LSTM bidirectional post-processing:
// Given out: [B, T, H2], take last timestep out[:, T-1, :] -> [B, H2]
// then compute linear: y = last @ W^T + b
// W: [O, H2], b: [O] (optional), y: [B, O]
//
// float32 only, contiguous, batch_first=True.
__global__ void last_timestep_linear_f32_kernel(
    const float* __restrict__ out_bth2,   // [B, T, H2]
    const float* __restrict__ W_oh2,      // [O, H2]
    const float* __restrict__ b_o,        // [O] or nullptr
    float* __restrict__ y_bo,             // [B, O]
    int B, int T, int H2, int O
) {
    int b = blockIdx.x;
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || o >= O) return;

    const int t_last = T - 1;
    const float* last = out_bth2 + ((int64_t)b * T + t_last) * (int64_t)H2;
    const float* wrow = W_oh2 + (int64_t)o * (int64_t)H2;

    float acc = 0.0f;
    // Naive dot; relies on compiler and L2 cache; still avoids materializing slice.
    for (int k = 0; k < H2; ++k) {
        acc = fmaf(last[k], wrow[k], acc);
    }
    if (b_o) acc += b_o[o];

    y_bo[(int64_t)b * O + o] = acc;
}

torch::Tensor last_timestep_linear_f32_cuda(
    torch::Tensor out,                   // [B, T, H2]
    torch::Tensor W,                     // [O, H2]
    c10::optional<torch::Tensor> b_opt   // [O]
) {
    CHECK_INPUT(out);
    CHECK_INPUT(W);

    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(W.scalar_type() == at::kFloat, "W must be float32");

    TORCH_CHECK(out.dim() == 3, "out must be 3D [B, T, H2]");
    TORCH_CHECK(W.dim() == 2, "W must be 2D [O, H2]");
    TORCH_CHECK(out.size(2) == W.size(1), "H2 mismatch: out.size(2) must equal W.size(1)");
    TORCH_CHECK(out.size(1) >= 1, "T must be >= 1");

    const int B = (int)out.size(0);
    const int T = (int)out.size(1);
    const int H2 = (int)out.size(2);
    const int O = (int)W.size(0);

    const float* b_ptr = nullptr;
    torch::Tensor b;
    if (b_opt.has_value()) {
        b = b_opt.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == O, "bias must be [O]");
        b_ptr = (const float*)b.data_ptr<float>();
    }

    auto y = torch::empty({B, O}, out.options());

    const int threads = 256;
    dim3 blocks(B, (O + threads - 1) / threads);

    c10::cuda::CUDAGuard device_guard(out.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    last_timestep_linear_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)out.data_ptr<float>(),
        (const float*)W.data_ptr<float>(),
        b_ptr,
        (float*)y.data_ptr<float>(),
        B, T, H2, O
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor last_timestep_linear_f32_cuda(
    torch::Tensor out,
    torch::Tensor W,
    c10::optional<torch::Tensor> b_opt
);
"""

_ext_name = "custom_ops_lib_lstm_bidirectional"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["last_timestep_linear_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    Bidirectional multi-layer LSTM (kept on PyTorch/cuDNN) with a fused custom CUDA kernel
    for last-timestep extraction + final Linear.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        out, _hn = self.lstm(x, (h0, c0))  # out: [B, T, 2H]
        W = self.fc.weight
        b = self.fc.bias

        # Narrow dispatch to preserve correctness and avoid silent layout issues.
        if (
            out.is_cuda and out.dtype == torch.float32 and out.is_contiguous() and out.dim() == 3
            and W.is_cuda and W.dtype == torch.float32 and W.is_contiguous() and W.dim() == 2
            and (b is None or (b.is_cuda and b.dtype == torch.float32 and b.is_contiguous() and b.dim() == 1))
        ):
            return self.custom_ops_lib.last_timestep_linear_f32_cuda(out, W, b)

        # Fallback
        return self.fc(out[:, -1, :])