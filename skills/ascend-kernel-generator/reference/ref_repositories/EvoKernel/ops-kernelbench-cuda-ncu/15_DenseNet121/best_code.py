import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------
# Custom CUDA extensions
# -----------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif

static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ float relu_f(float v) { return v > 0.f ? v : 0.f; }

// -----------------------
// Vectorized ReLU inplace
// -----------------------
__global__ __launch_bounds__(256, 2)
void relu_inplace_f32_vec(float* __restrict__ x, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Prefer float4 when 16B aligned.
    if ((((uintptr_t)x) & 15u) == 0u) {
        int64_t n4 = n >> 2;
        float4* x4 = reinterpret_cast<float4*>(x);
        for (int64_t i = tid; i < n4; i += stride) {
            float4 v = x4[i];
            v.x = relu_f(v.x);
            v.y = relu_f(v.y);
            v.z = relu_f(v.z);
            v.w = relu_f(v.w);
            x4[i] = v;
        }
        int64_t base = n4 << 2;
        for (int64_t i = base + tid; i < n; i += stride) {
            x[i] = relu_f(x[i]);
        }
    } else {
        for (int64_t i = tid; i < n; i += stride) {
            x[i] = relu_f(x[i]);
        }
    }
}

__global__ __launch_bounds__(256, 2)
void relu_inplace_f16_vec(__half* __restrict__ x, int64_t n) {
#if __CUDA_ARCH__ >= 530
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Prefer half2 when 4B aligned.
    if ((((uintptr_t)x) & 3u) == 0u) {
        int64_t n2 = n >> 1;
        half2* x2 = reinterpret_cast<half2*>(x);
        half2 z = __float2half2_rn(0.0f);
        for (int64_t i = tid; i < n2; i += stride) {
            half2 v = x2[i];
            x2[i] = __hmax2(v, z);
        }
        int64_t base = n2 << 1;
        for (int64_t i = base + tid; i < n; i += stride) {
            __half v = x[i];
            x[i] = __hlt(v, __float2half(0.0f)) ? __float2half(0.0f) : v;
        }
    } else {
        for (int64_t i = tid; i < n; i += stride) {
            __half v = x[i];
            x[i] = __hlt(v, __float2half(0.0f)) ? __float2half(0.0f) : v;
        }
    }
#else
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        float fv = __half2float(x[i]);
        x[i] = __float2half(fv > 0.f ? fv : 0.f);
    }
#endif
}

torch::Tensor relu_forward_inplace_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.is_floating_point(), "x must be floating point");
    const auto n = x.numel();
    if (n == 0) return x;

    const int threads = 256;
    int blocks = (int)ceil_div_i64(n, threads);
    // Cap blocks to a reasonable number to reduce launch overhead while preserving MLP.
    if (blocks > 8192) blocks = 8192;
    if (blocks < 1) blocks = 1;

    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (x.scalar_type() == at::ScalarType::Float) {
        relu_inplace_f32_vec<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (x.scalar_type() == at::ScalarType::Half) {
        relu_inplace_f16_vec<<<blocks, threads, 0, stream>>>((__half*)x.data_ptr<at::Half>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(false, "unsupported dtype for relu_forward_inplace_cuda");
    }
    return x;
}

// -----------------------
// Correct NCHW cat along channels: y = cat(a,b) on dim=1
// Strategy: copy per (n,c) plane of HW elements to correct destination,
// with vectorized inner copy when aligned. This avoids layout bugs from "flat" copies.
// -----------------------
__global__ __launch_bounds__(256, 2)
void cat_nchw_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Ca, int Cb, int H, int W
) {
    int hw = H * W;
    int total_planes = N * (Ca + Cb); // each plane is HW floats

    int plane = (int)blockIdx.y;
    if (plane >= total_planes) return;

    int n = plane / (Ca + Cb);
    int c = plane - n * (Ca + Cb);

    const float* src;
    if (c < Ca) src = a + ((n * Ca + c) * (int64_t)hw);
    else        src = b + ((n * Cb + (c - Ca)) * (int64_t)hw);

    float* dst = y + ((n * (Ca + Cb) + c) * (int64_t)hw);

    int tid = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int stride = (int)blockDim.x * gridDim.x;

    // Try float4 copy if both pointers 16B aligned and hw multiple of 4.
    bool aligned16 = ((((uintptr_t)src) | ((uintptr_t)dst)) & 15u) == 0u;

    if (aligned16) {
        int hw4 = hw >> 2;
        const float4* s4 = reinterpret_cast<const float4*>(src);
        float4* d4 = reinterpret_cast<float4*>(dst);
        for (int i = tid; i < hw4; i += stride) d4[i] = s4[i];
        int base = hw4 << 2;
        for (int i = base + tid; i < hw; i += stride) dst[i] = src[i];
    } else {
        for (int i = tid; i < hw; i += stride) dst[i] = src[i];
    }
}

__global__ __launch_bounds__(256, 2)
void cat_nchw_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ y,
    int N, int Ca, int Cb, int H, int W
) {
    int hw = H * W;
    int total_planes = N * (Ca + Cb);

    int plane = (int)blockIdx.y;
    if (plane >= total_planes) return;

    int n = plane / (Ca + Cb);
    int c = plane - n * (Ca + Cb);

    const __half* src;
    if (c < Ca) src = a + ((n * Ca + c) * (int64_t)hw);
    else        src = b + ((n * Cb + (c - Ca)) * (int64_t)hw);

    __half* dst = y + ((n * (Ca + Cb) + c) * (int64_t)hw);

    int tid = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int stride = (int)blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
    bool aligned4 = ((((uintptr_t)src) | ((uintptr_t)dst)) & 3u) == 0u;
    if (aligned4) {
        int hw2 = hw >> 1;
        const half2* s2 = reinterpret_cast<const half2*>(src);
        half2* d2 = reinterpret_cast<half2*>(dst);
        for (int i = tid; i < hw2; i += stride) d2[i] = s2[i];
        int base = hw2 << 1;
        for (int i = base + tid; i < hw; i += stride) dst[i] = src[i];
    } else {
        for (int i = tid; i < hw; i += stride) dst[i] = src[i];
    }
#else
    for (int i = tid; i < hw; i += stride) dst[i] = src[i];
#endif
}

torch::Tensor cat_channels_forward_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_CUDA(a); CHECK_CUDA(b);
    CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b);
    TORCH_CHECK(a.is_floating_point() && b.is_floating_point(), "a and b must be floating point");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have same dtype");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "a and b must be NCHW tensors");
    TORCH_CHECK(a.size(0) == b.size(0), "N mismatch");
    TORCH_CHECK(a.size(2) == b.size(2) && a.size(3) == b.size(3), "H/W mismatch");

    // Stronger contract: require standard contiguous NCHW strides.
    TORCH_CHECK(a.stride(3) == 1 && a.stride(2) == a.size(3), "a must be contiguous NCHW");
    TORCH_CHECK(b.stride(3) == 1 && b.stride(2) == b.size(3), "b must be contiguous NCHW");

    int N = (int)a.size(0);
    int Ca = (int)a.size(1);
    int Cb = (int)b.size(1);
    int H = (int)a.size(2);
    int W = (int)a.size(3);

    auto y = torch::empty({N, Ca + Cb, H, W}, a.options());

    const at::cuda::CUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // 2D grid: x dimension advances within HW plane; y dimension selects (n,c) plane.
    const int threads = 256;
    int hw = H * W;
    // Use enough blocks in x to cover hw with grid-stride and add some MLP.
    int blocks_x = (int)ceil_div_i64(hw, threads);
    if (blocks_x < 1) blocks_x = 1;
    if (blocks_x > 64) blocks_x = 64; // avoid too many blocks per plane
    dim3 grid(blocks_x, (unsigned)(N * (Ca + Cb)), 1);

    if (a.scalar_type() == at::ScalarType::Float) {
        cat_nchw_f32<<<grid, threads, 0, stream>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(),
            N, Ca, Cb, H, W
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (a.scalar_type() == at::ScalarType::Half) {
        cat_nchw_f16<<<grid, threads, 0, stream>>>(
            (const __half*)a.data_ptr<at::Half>(), (const __half*)b.data_ptr<at::Half>(), (__half*)y.data_ptr<at::Half>(),
            N, Ca, Cb, H, W
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(false, "unsupported dtype for cat_channels_forward_cuda");
    }
    return y;
}
"""

cpp_src = r"""
torch::Tensor relu_forward_inplace_cuda(torch::Tensor x);
torch::Tensor cat_channels_forward_cuda(torch::Tensor a, torch::Tensor b);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_dense_net121_cat_relu_v5_nchw_safe",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "relu_forward_inplace_cuda",
        "cat_channels_forward_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -----------------------
# Model rewrite
# -----------------------

def _ensure_contig_nchw(x: torch.Tensor) -> torch.Tensor:
    # Enforce standard contiguous NCHW to match custom kernels' contract.
    # This avoids silent layout bugs (e.g., channels_last) at the cost of occasional copies.
    if x.is_contiguous() and x.dim() == 4 and x.stride(3) == 1 and x.stride(2) == x.size(3):
        return x
    return x.contiguous()

class DenseLayerNew(nn.Module):
    def __init__(self, in_features: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_p = 0.0

    def forward(self, x):
        y = self.bn(x)
        y = _ensure_contig_nchw(y)
        if y.is_cuda:
            custom_ops_lib.relu_forward_inplace_cuda(y)
        else:
            y = F.relu(y, inplace=True)
        y = self.conv(y)
        if self.drop_p != 0.0 and self.training:
            y = F.dropout(y, p=self.drop_p, training=True)
        return y

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DenseLayerNew(num_input_features + i * growth_rate, growth_rate) for i in range(num_layers)]
        )

    def forward(self, x):
        x = _ensure_contig_nchw(x)
        for layer in self.layers:
            new_feature = layer(x)
            new_feature = _ensure_contig_nchw(new_feature)
            if x.is_cuda:
                x = custom_ops_lib.cat_channels_forward_cuda(x, new_feature)
            else:
                x = torch.cat([x, new_feature], dim=1)
            # output of cat is contiguous by construction
        return x

class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.bn(x)
        y = _ensure_contig_nchw(y)
        if y.is_cuda:
            custom_ops_lib.relu_forward_inplace_cuda(y)
        else:
            y = F.relu(y, inplace=True)
        y = self.conv(y)
        y = self.pool(y)
        return y

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super().__init__()

        self.stem_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, nlayers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=nlayers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + nlayers * growth_rate

            if i != len(block_layers) - 1:
                trans = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(trans)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = _ensure_contig_nchw(x)
        if x.is_cuda:
            custom_ops_lib.relu_forward_inplace_cuda(x)
        else:
            x = F.relu(x, inplace=True)
        x = self.stem_pool(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = _ensure_contig_nchw(x)
        if x.is_cuda:
            custom_ops_lib.relu_forward_inplace_cuda(x)
        else:
            x = F.relu(x, inplace=True)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x