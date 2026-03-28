import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA: (x * scale[c]) -> tanh -> (* bias[c]) -> sigmoid
# x: [N, C, D, H, W] contiguous (NCDHW) float32 CUDA
# scale/bias: [C,1,1,1] contiguous float32 CUDA (broadcast per channel)
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float sigmoid_f32(float x) {
    // expf is fast under --use_fast_math
    return 1.0f / (1.0f + expf(-x));
}

// Keep math in separate function to help compiler scheduling
__device__ __forceinline__ float fused_op(float v, float sc, float bc) {
    v = tanhf(v * sc);
    v = sigmoid_f32(v * bc);
    return v;
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_chw_f32_vec4(
    const float* __restrict__ x,       // [N*C*inner]
    const float* __restrict__ scale_c, // [C]
    const float* __restrict__ bias_c,  // [C]
    float* __restrict__ out,           // [N*C*inner]
    int C,
    int inner,
    int NC
) {
    int c = (int)blockIdx.y;
    if (c >= C) return;

    // per-channel params (read-only cache)
    float sc = __ldg(&scale_c[c]);
    float bc = __ldg(&bias_c[c]);

    // work decomposition:
    // for each channel, iterate nc = c, c+C, ... (so nc maps to (n,c))
    // for each nc, stream inner with vectorized grid-stride
    int tid = (int)threadIdx.x;

    int vec_inner4 = inner >> 2; // inner/4
    int idx4 = (int)blockIdx.x * THREADS + tid;
    int stride4 = (int)gridDim.x * THREADS;

    for (int nc = c; nc < NC; nc += C) {
        int base = nc * inner;

        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
        float4* __restrict__ o4 = reinterpret_cast<float4*>(out + base);

        for (int i4 = idx4; i4 < vec_inner4; i4 += stride4) {
            float4 v = x4[i4];

            // Unroll the 4 lanes explicitly for better scheduling
            float o0 = fused_op(v.x, sc, bc);
            float o1 = fused_op(v.y, sc, bc);
            float o2 = fused_op(v.z, sc, bc);
            float o3 = fused_op(v.w, sc, bc);

            o4[i4] = make_float4(o0, o1, o2, o3);
        }
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_chw_f32_vec2(
    const float* __restrict__ x,
    const float* __restrict__ scale_c,
    const float* __restrict__ bias_c,
    float* __restrict__ out,
    int C,
    int inner,
    int NC
) {
    int c = (int)blockIdx.y;
    if (c >= C) return;

    float sc = __ldg(&scale_c[c]);
    float bc = __ldg(&bias_c[c]);

    int tid = (int)threadIdx.x;
    int vec_inner2 = inner >> 1; // inner/2
    int idx2 = (int)blockIdx.x * THREADS + tid;
    int stride2 = (int)gridDim.x * THREADS;

    for (int nc = c; nc < NC; nc += C) {
        int base = nc * inner;

        const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x + base);
        float2* __restrict__ o2 = reinterpret_cast<float2*>(out + base);

        for (int i2 = idx2; i2 < vec_inner2; i2 += stride2) {
            float2 v = x2[i2];
            float a = fused_op(v.x, sc, bc);
            float b = fused_op(v.y, sc, bc);
            o2[i2] = make_float2(a, b);
        }
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_chw_f32_scalar(
    const float* __restrict__ x,
    const float* __restrict__ scale_c,
    const float* __restrict__ bias_c,
    float* __restrict__ out,
    int C,
    int inner,
    int64_t NC64
) {
    int c = (int)blockIdx.y;
    if (c >= C) return;

    float sc = __ldg(&scale_c[c]);
    float bc = __ldg(&bias_c[c]);

    int tid = (int)threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * (int64_t)THREADS + tid;
    int64_t stride = (int64_t)gridDim.x * (int64_t)THREADS;

    for (int64_t nc = c; nc < NC64; nc += C) {
        int64_t base = nc * (int64_t)inner;
        for (int64_t i = idx; i < (int64_t)inner; i += stride) {
            float v = x[base + i];
            out[base + i] = fused_op(v, sc, bc);
        }
    }
}

static inline bool is_aligned(uintptr_t p, uintptr_t a) {
    return (p & (a - 1)) == 0;
}

torch::Tensor fused_conv3d_scaling_tanh_multiply_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor scaling_factor,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(scaling_factor.is_cuda(), "scaling_factor must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(scaling_factor.scalar_type() == torch::kFloat32, "scaling_factor must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCDHW)");
    TORCH_CHECK(scaling_factor.is_contiguous(), "scaling_factor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(scaling_factor.dim() == 4, "scaling_factor must be 4D [C,1,1,1]");
    TORCH_CHECK(bias.dim() == 4, "bias must be 4D [C,1,1,1]");

    int64_t N = x.size(0);
    int64_t C64 = x.size(1);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    TORCH_CHECK(scaling_factor.size(0) == C64, "scaling_factor.size(0) must match x.size(1)");
    TORCH_CHECK(bias.size(0) == C64, "bias.size(0) must match x.size(1)");
    TORCH_CHECK(scaling_factor.size(1) == 1 && scaling_factor.size(2) == 1 && scaling_factor.size(3) == 1,
                "scaling_factor must have shape [C,1,1,1]");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1 && bias.size(3) == 1,
                "bias must have shape [C,1,1,1]");

    TORCH_CHECK(C64 <= INT_MAX, "C too large");
    int C = (int)C64;

    int64_t inner64 = D * H * W;
    TORCH_CHECK(inner64 > 0, "invalid inner");
    TORCH_CHECK(inner64 <= INT_MAX, "inner too large for fast path");
    int inner = (int)inner64;

    int64_t NC64 = N * C64;
    TORCH_CHECK(NC64 > 0, "invalid NC");
    TORCH_CHECK(NC64 <= (int64_t)INT_MAX, "NC too large for this kernel");
    int NC = (int)NC64;

    auto scale_c = scaling_factor.view({C});
    auto bias_c  = bias.view({C});

    auto out = torch::empty_like(x);

    const float* xptr = x.data_ptr<float>();
    float* outptr = out.data_ptr<float>();

    // grid.x chosen to cover inner; capped to avoid huge grids; we will grid-stride anyway
    // Use slightly smaller caps than extreme to reduce launch overhead while keeping enough CTAs.
    constexpr int THREADS = 256;
    int blocks_x = (inner + (THREADS - 1)) / THREADS;
    if (blocks_x < 1) blocks_x = 1;
    if (blocks_x > 2048) blocks_x = 2048;

    dim3 block(THREADS, 1, 1);
    dim3 grid(blocks_x, (unsigned)C, 1);

    // Host-side alignment/path selection
    uintptr_t xp = (uintptr_t)xptr;
    uintptr_t op = (uintptr_t)outptr;

    bool aligned16 = is_aligned(xp, 16) && is_aligned(op, 16);
    bool aligned8  = is_aligned(xp, 8)  && is_aligned(op, 8);

    if (aligned16 && ((inner & 3) == 0)) {
        fused_chw_f32_vec4<THREADS><<<grid, block>>>(
            xptr,
            scale_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            outptr,
            C, inner, NC
        );
    } else if (aligned8 && ((inner & 1) == 0)) {
        fused_chw_f32_vec2<THREADS><<<grid, block>>>(
            xptr,
            scale_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            outptr,
            C, inner, NC
        );
    } else {
        fused_chw_f32_scalar<THREADS><<<grid, block>>>(
            xptr,
            scale_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            outptr,
            C, inner, (int64_t)NC
        );
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor fused_conv3d_scaling_tanh_multiply_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor scaling_factor,
    torch::Tensor bias
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_scaling_tanh_mul_sigmoid_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_conv3d_scaling_tanh_multiply_sigmoid_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv3d computed by PyTorch/cuDNN, then a fused CUDA kernel applies:
      (x * scaling_factor) -> tanh -> (* bias) -> sigmoid

    Fast-path only for CUDA float32 contiguous tensors with scaling_factor/bias shaped [C,1,1,1];
    otherwise falls back to PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or x.dtype != torch.float32:
            x = x * self.scaling_factor
            x = torch.tanh(x)
            x = x * self.bias
            x = torch.sigmoid(x)
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        scale = self.scaling_factor
        bias = self.bias

        if (not scale.is_cuda) or scale.device != x.device:
            scale = scale.to(device=x.device)
        if (not bias.is_cuda) or bias.device != x.device:
            bias = bias.to(device=x.device)

        if scale.dtype != torch.float32:
            scale = scale.float()
        if bias.dtype != torch.float32:
            bias = bias.float()

        if not scale.is_contiguous():
            scale = scale.contiguous()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        if (
            scale.dim() != 4 or bias.dim() != 4 or
            scale.size(1) != 1 or scale.size(2) != 1 or scale.size(3) != 1 or
            bias.size(1) != 1 or bias.size(2) != 1 or bias.size(3) != 1 or
            scale.size(0) != x.size(1) or bias.size(0) != x.size(1)
        ):
            x = x * self.scaling_factor
            x = torch.tanh(x)
            x = x * self.bias
            x = torch.sigmoid(x)
            return x

        return self.custom_ops_lib.fused_conv3d_scaling_tanh_multiply_sigmoid_cuda(x, scale, bias)


# Keep original helper signatures for integration consistency
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]