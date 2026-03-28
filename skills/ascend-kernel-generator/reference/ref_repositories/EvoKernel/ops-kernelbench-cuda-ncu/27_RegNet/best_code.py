import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA extension: fused GlobalAveragePool (mean over H,W) + Linear
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

// Fused: y[n,o] = bias[o] + sum_c( mean_{h,w}(x[n,c,h,w]) * w[o,c] )
//
// Design goals:
// - Match PyTorch semantics: mean uses division by (H*W), FP32 accumulation.
// - Avoid previous failure modes: no constant-memory weight caching, no risky float4 loads.
// - Keep logic simple and correct; performance comes from eliminating intermediate tensor
//   (GAP output) and reducing launches.
//
// Kernel mapping:
// - Each block computes one (n, o).
// - Threads iterate over channels c with stride blockDim.x.
// - For each c: threads cooperatively reduce over HW with a strided loop.
// - Accumulate partial dot contributions in FP32 and reduce within block.
__global__ void gap_linear_fwd_kernel(
    const float* __restrict__ x,   // [N,C,H,W] contiguous NCHW
    const float* __restrict__ w,   // [O,C] contiguous
    const float* __restrict__ b,   // [O] contiguous (can be nullptr)
    float* __restrict__ y,         // [N,O]
    int N, int C, int H, int W, int O
) {
    int n = (int)blockIdx.y;
    int o = (int)blockIdx.x;
    if (n >= N || o >= O) return;

    int tid = (int)threadIdx.x;
    int HW = H * W;
    float invHW = 1.0f / (float)HW;

    const float* __restrict__ x_n = x + ( (n * C) * HW );
    const float* __restrict__ w_o = w + ( (o * C) );

    float acc = 0.0f;

    // Iterate channels assigned to this thread
    for (int c = tid; c < C; c += (int)blockDim.x) {
        const float* __restrict__ x_nc = x_n + (c * HW);

        float sum = 0.0f;
        // Reduce over HW
        for (int i = 0; i < HW; ++i) {
            sum += x_nc[i];
        }
        float mean = sum * invHW;
        acc = fmaf(mean, w_o[c], acc);
    }

    // Block reduce acc
    extern __shared__ float sh[];
    sh[tid] = acc;
    __syncthreads();

    for (int s = ((int)blockDim.x >> 1); s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float out = sh[0];
        if (b != nullptr) out += b[o];
        y[n * O + o] = out;
    }
}

torch::Tensor gap_linear_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    CHECK_CUDA(x); CHECK_CUDA(w);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w);
    CHECK_FLOAT(x); CHECK_FLOAT(w);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 2, "w must be [O, C]");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int O = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == C, "w.size(1) must match x.size(1)");

    const float* bptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        CHECK_CUDA(b);
        CHECK_CONTIGUOUS(b);
        CHECK_FLOAT(b);
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == O, "b must be [O]");
        bptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N, O}, x.options());

    // Choose block size as power-of-two up to 256 for reduction efficiency.
    int block = 256;
    if (C < 256) {
        // round up to next power of two, minimum 32
        int p = 32;
        while (p < C) p <<= 1;
        block = p;
        if (block > 256) block = 256;
    }

    dim3 grid(O, N, 1);
    size_t shmem = (size_t)block * sizeof(float);
    gap_linear_fwd_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, C, H, W, O
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor gap_linear_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_regnet_gap_linear_fused_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gap_linear_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -----------------------------------------------------------------------------
# ModelNew: same feature extractor; fused GAP+FC in forward
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()
        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        self.feature_extractor = nn.Sequential(*layers)

        self.fc = nn.Linear(block_widths[-1], output_classes)

    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        # Use fused CUDA op when safe; otherwise fall back to PyTorch path.
        if x.is_cuda and x.dtype == torch.float32 and (not self.training):
            x = x.contiguous()
            w = self.fc.weight.contiguous()  # [O,C]
            b = self.fc.bias
            if b is None:
                b = torch.empty(0, device=x.device, dtype=x.dtype)
            else:
                b = b.contiguous()
            return custom_ops_lib.gap_linear_forward_cuda(x, w, b)

        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x