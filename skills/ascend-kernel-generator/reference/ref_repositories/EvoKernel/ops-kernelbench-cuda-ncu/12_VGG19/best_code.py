import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# CUDA extension:
#   (1) in-place ReLU (scalar + float4)
#   (2) fused ReLU + MaxPool2d(2,2) for channels-last (NHWC memory)
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif

#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

#define RELU_F32(v) fmaxf((v), 0.0f)

__global__ __launch_bounds__(256, 2)
void relu_inplace_f32_scalar(float* __restrict__ x, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    #pragma unroll 1
    for (int64_t i = tid; i < n; i += stride * 4) {
        int64_t j0 = i;
        int64_t j1 = i + stride;
        int64_t j2 = i + stride * 2;
        int64_t j3 = i + stride * 3;

        if (j0 < n) { float v = x[j0]; x[j0] = RELU_F32(v); }
        if (j1 < n) { float v = x[j1]; x[j1] = RELU_F32(v); }
        if (j2 < n) { float v = x[j2]; x[j2] = RELU_F32(v); }
        if (j3 < n) { float v = x[j3]; x[j3] = RELU_F32(v); }
    }
}

__global__ __launch_bounds__(256, 2)
void relu_inplace_f32_vec4(float4* __restrict__ x4, int64_t n4) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    #pragma unroll 1
    for (int64_t i = tid; i < n4; i += stride * 2) {
        int64_t i0 = i;
        int64_t i1 = i + stride;

        if (i0 < n4) {
            float4 v = x4[i0];
            v.x = RELU_F32(v.x);
            v.y = RELU_F32(v.y);
            v.z = RELU_F32(v.z);
            v.w = RELU_F32(v.w);
            x4[i0] = v;
        }
        if (i1 < n4) {
            float4 v = x4[i1];
            v.x = RELU_F32(v.x);
            v.y = RELU_F32(v.y);
            v.z = RELU_F32(v.z);
            v.w = RELU_F32(v.w);
            x4[i1] = v;
        }
    }
}

// ---------------- Fused ReLU + MaxPool2d(2,2) for NHWC (channels-last) ----------------
//
// Logical shape is NCHW, but memory is NHWC when channels_last contiguous.
// We operate on NHWC memory order directly.
//
// Grid mapping:
//   grid.x -> channel packs (C4) for vec4, or channels (C) for scalar
//   grid.y -> w2 in [0, Wo)
//   grid.z -> zh in [0, N*Ho), decode n = zh/Ho, h2 = zh%Ho
//

__global__ __launch_bounds__(256, 2)
void relu_maxpool2d_nhwc_f32_vec4(
    const float4* __restrict__ in4,
    float4* __restrict__ out4,
    int N, int H, int W, int C4
) {
    int c4 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c4 >= C4) return;

    int Ho = H >> 1;
    int Wo = W >> 1;

    int w2 = (int)blockIdx.y;
    int zh = (int)blockIdx.z;
    int n = zh / Ho;
    int h2 = zh - n * Ho;

    int h0 = h2 << 1;
    int w0 = w2 << 1;

    int64_t base00 = (((int64_t)n * H + h0) * W + w0) * (int64_t)C4 + c4;
    int64_t base01 = base00 + (int64_t)C4;
    int64_t base10 = base00 + (int64_t)W * C4;
    int64_t base11 = base10 + (int64_t)C4;

    float4 a = in4[base00];
    float4 b = in4[base01];
    float4 c = in4[base10];
    float4 d = in4[base11];

    a.x = RELU_F32(a.x); a.y = RELU_F32(a.y); a.z = RELU_F32(a.z); a.w = RELU_F32(a.w);
    b.x = RELU_F32(b.x); b.y = RELU_F32(b.y); b.z = RELU_F32(b.z); b.w = RELU_F32(b.w);
    c.x = RELU_F32(c.x); c.y = RELU_F32(c.y); c.z = RELU_F32(c.z); c.w = RELU_F32(c.w);
    d.x = RELU_F32(d.x); d.y = RELU_F32(d.y); d.z = RELU_F32(d.z); d.w = RELU_F32(d.w);

    float4 m;
    m.x = fmaxf(fmaxf(a.x, b.x), fmaxf(c.x, d.x));
    m.y = fmaxf(fmaxf(a.y, b.y), fmaxf(c.y, d.y));
    m.z = fmaxf(fmaxf(a.z, b.z), fmaxf(c.z, d.z));
    m.w = fmaxf(fmaxf(a.w, b.w), fmaxf(c.w, d.w));

    int64_t oidx = (((int64_t)n * Ho + h2) * Wo + w2) * (int64_t)C4 + c4;
    out4[oidx] = m;
}

__global__ __launch_bounds__(256, 2)
void relu_maxpool2d_nhwc_f32_scalar(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N, int H, int W, int C
) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= C) return;

    int Ho = H >> 1;
    int Wo = W >> 1;

    int w2 = (int)blockIdx.y;
    int zh = (int)blockIdx.z;
    int n = zh / Ho;
    int h2 = zh - n * Ho;

    int h0 = h2 << 1;
    int w0 = w2 << 1;

    int64_t idx00 = (((int64_t)n * H + h0) * W + w0) * (int64_t)C + c;
    int64_t idx01 = idx00 + (int64_t)C;
    int64_t idx10 = idx00 + (int64_t)W * C;
    int64_t idx11 = idx10 + (int64_t)C;

    float v00 = RELU_F32(in[idx00]);
    float v01 = RELU_F32(in[idx01]);
    float v10 = RELU_F32(in[idx10]);
    float v11 = RELU_F32(in[idx11]);

    float m0 = v00 > v01 ? v00 : v01;
    float m1 = v10 > v11 ? v10 : v11;
    float m  = m0 > m1 ? m0 : m1;

    int64_t oidx = (((int64_t)n * Ho + h2) * Wo + w2) * (int64_t)C + c;
    out[oidx] = m;
}

torch::Tensor relu_inplace_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.is_contiguous(), "relu_inplace_cuda expects contiguous tensor");
    TORCH_CHECK(x.numel() > 0, "x must have numel > 0");

    const c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    int64_t n = x.numel();
    float* ptr = x.data_ptr<float>();

    constexpr int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 131072 ? 131072 : blocks;
    blocks = blocks < 1 ? 1 : blocks;

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    bool aligned16 = (addr & 0xF) == 0;

    if (aligned16 && ((n & 3) == 0)) {
        int64_t n4 = n >> 2;
        int blocks4 = (int)((n4 + threads - 1) / threads);
        blocks4 = blocks4 > 131072 ? 131072 : blocks4;
        blocks4 = blocks4 < 1 ? 1 : blocks4;
        relu_inplace_f32_vec4<<<blocks4, threads, 0, stream>>>(
            reinterpret_cast<float4*>(ptr), n4
        );
    } else {
        relu_inplace_f32_scalar<<<blocks, threads, 0, stream>>>(ptr, n);
    }
    return x;
}

torch::Tensor relu_maxpool2d_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 4, "relu_maxpool2d_cuda expects 4D tensor");
    TORCH_CHECK(x.size(2) % 2 == 0 && x.size(3) % 2 == 0, "H and W must be even");
    TORCH_CHECK(
        x.is_contiguous(at::MemoryFormat::ChannelsLast),
        "relu_maxpool2d_cuda expects channels-last contiguous tensor"
    );

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int Ho = H >> 1;
    const int Wo = W >> 1;

    const c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto out = torch::empty({N, C, Ho, Wo}, x.options().memory_format(at::MemoryFormat::ChannelsLast));

    const float* in = x.data_ptr<float>();
    float* o = out.data_ptr<float>();

    constexpr int threads = 256;

    bool c4_ok = ((C & 3) == 0);
    uintptr_t ain = reinterpret_cast<uintptr_t>(in);
    uintptr_t aout = reinterpret_cast<uintptr_t>(o);
    bool aligned16 = ((ain & 0xF) == 0) && ((aout & 0xF) == 0);

    if (c4_ok && aligned16) {
        int C4 = C >> 2;
        dim3 block(threads, 1, 1);
        dim3 grid((C4 + threads - 1) / threads, Wo, N * Ho);
        relu_maxpool2d_nhwc_f32_vec4<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(in),
            reinterpret_cast<float4*>(o),
            N, H, W, C4
        );
    } else {
        dim3 block(threads, 1, 1);
        dim3 grid((C + threads - 1) / threads, Wo, N * Ho);
        relu_maxpool2d_nhwc_f32_scalar<<<grid, block, 0, stream>>>(
            in, o, N, H, W, C
        );
    }
    return out;
}
"""

cpp_source = r"""
torch::Tensor relu_inplace_cuda(torch::Tensor x);
torch::Tensor relu_maxpool2d_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_vgg19_fused_pool_relu_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_inplace_cuda", "relu_maxpool2d_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # VGG19 conv stack (keep as standard Conv2d so cuDNN can pick optimal kernels)
        self.features = nn.ModuleList([
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes),
        )

    def _relu_inplace(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom in-place ReLU only when contiguous in current format.
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return custom_ops_lib.relu_inplace_cuda(x)
        return torch.relu_(x)

    def _relu_pool(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.is_contiguous(memory_format=torch.channels_last)
            and (x.size(2) % 2 == 0)
            and (x.size(3) % 2 == 0)
        ):
            return custom_ops_lib.relu_maxpool2d_cuda(x)
        return torch.max_pool2d(torch.relu(x), kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep activations channels-last so fused pooling triggers and cuDNN can use NHWC paths.
        if x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous(memory_format=torch.channels_last)

        it = iter(self.features)

        # Block 1
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_pool(x)

        # Block 2
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_pool(x)

        # Block 3
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_pool(x)

        # Block 4
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_pool(x)

        # Block 5
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_inplace(x)
        x = next(it)(x); x = self._relu_pool(x)

        # Flatten + FC (make contiguous NCHW/row-major for Linear)
        x = x.contiguous(memory_format=torch.contiguous_format)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x