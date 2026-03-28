import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: (1) in-place ReLU (kept for FC layers)
#              (2) fused ReLU + MaxPool2d(2,2) for feature blocks
#                  - NHWC (channels-last) float4 vectorized fast-path
#                  - NCHW scalar fallback
# ------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

#define RELU_F32(v) fmaxf((v), 0.0f)

__global__ __launch_bounds__(256, 2) void relu_inplace_f32_vec4(float4* __restrict__ x4, int64_t n4) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    #pragma unroll 2
    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        v.x = RELU_F32(v.x);
        v.y = RELU_F32(v.y);
        v.z = RELU_F32(v.z);
        v.w = RELU_F32(v.w);
        x4[i] = v;
    }
}

__global__ __launch_bounds__(256, 2) void relu_inplace_f32_scalar(float* __restrict__ x, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    #pragma unroll 4
    for (int64_t i = tid; i < n; i += stride) {
        float v = x[i];
        x[i] = RELU_F32(v);
    }
}

// Fused ReLU + MaxPool2x2 stride2
// NHWC: input [N,H,W,C] contiguous, output [N,H/2,W/2,C] contiguous.
// Each thread processes one (n, oh, ow, c_vec4) and loads 4 float4s (2x2 window).
__global__ __launch_bounds__(256, 2)
void relu_maxpool2x2_s2_nhwc_vec4(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int N, int H, int W, int C4
) {
    int OH = H >> 1;
    int OW = W >> 1;
    int64_t total = (int64_t)N * OH * OW * C4;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total; idx += stride) {
        int64_t t = idx;
        int c4 = (int)(t % C4); t /= C4;
        int ow = (int)(t % OW); t /= OW;
        int oh = (int)(t % OH); t /= OH;
        int n  = (int)t;

        int ih = oh << 1;
        int iw = ow << 1;

        // Base offsets in float4 units: (((n*H + ih)*W + iw)*C4 + c4)
        int64_t base00 = (((int64_t)n * H + ih) * W + iw) * C4 + c4;
        int64_t base01 = base00 + C4;              // (ih, iw+1)
        int64_t base10 = base00 + (int64_t)W * C4; // (ih+1, iw)
        int64_t base11 = base10 + C4;              // (ih+1, iw+1)

        float4 v00 = in[base00];
        float4 v01 = in[base01];
        float4 v10 = in[base10];
        float4 v11 = in[base11];

        // ReLU then maxpool across 2x2
        float4 r;
        float a0, a1, a2, a3;

        a0 = RELU_F32(v00.x); a0 = fmaxf(a0, RELU_F32(v01.x)); a0 = fmaxf(a0, RELU_F32(v10.x)); a0 = fmaxf(a0, RELU_F32(v11.x));
        a1 = RELU_F32(v00.y); a1 = fmaxf(a1, RELU_F32(v01.y)); a1 = fmaxf(a1, RELU_F32(v10.y)); a1 = fmaxf(a1, RELU_F32(v11.y));
        a2 = RELU_F32(v00.z); a2 = fmaxf(a2, RELU_F32(v01.z)); a2 = fmaxf(a2, RELU_F32(v10.z)); a2 = fmaxf(a2, RELU_F32(v11.z));
        a3 = RELU_F32(v00.w); a3 = fmaxf(a3, RELU_F32(v01.w)); a3 = fmaxf(a3, RELU_F32(v10.w)); a3 = fmaxf(a3, RELU_F32(v11.w));

        r.x = a0; r.y = a1; r.z = a2; r.w = a3;

        int64_t out_off = (((int64_t)n * OH + oh) * OW + ow) * C4 + c4;
        out[out_off] = r;
    }
}

// NCHW fallback: input [N,C,H,W], output [N,C,H/2,W/2]
__global__ __launch_bounds__(256, 2)
void relu_maxpool2x2_s2_nchw_scalar(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N, int C, int H, int W
) {
    int OH = H >> 1;
    int OW = W >> 1;
    int64_t total = (int64_t)N * C * OH * OW;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total; idx += stride) {
        int64_t t = idx;
        int ow = (int)(t % OW); t /= OW;
        int oh = (int)(t % OH); t /= OH;
        int c  = (int)(t % C);  t /= C;
        int n  = (int)t;

        int ih = oh << 1;
        int iw = ow << 1;

        int64_t base = ((int64_t)n * C + c) * H * W;
        int64_t off00 = base + (int64_t)ih * W + iw;
        int64_t off01 = off00 + 1;
        int64_t off10 = off00 + W;
        int64_t off11 = off10 + 1;

        float v00 = RELU_F32(in[off00]);
        float v01 = RELU_F32(in[off01]);
        float v10 = RELU_F32(in[off10]);
        float v11 = RELU_F32(in[off11]);

        float m0 = fmaxf(v00, v01);
        float m1 = fmaxf(v10, v11);
        float mv = fmaxf(m0, m1);

        int64_t out_off = ((int64_t)n * C + c) * OH * OW + (int64_t)oh * OW + ow;
        out[out_off] = mv;
    }
}

torch::Tensor relu_inplace_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
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

torch::Tensor relu_maxpool2x2_s2_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.numel() > 0, "x must have numel > 0");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    const c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // If channels-last contiguous, treat as NHWC and use vectorized path.
    if (x.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        // x sizes in NCHW notation but memory is NHWC (channels-last)
        int N = (int)x.size(0);
        int C = (int)x.size(1);
        int H = (int)x.size(2);
        int W = (int)x.size(3);
        TORCH_CHECK((H & 1) == 0 && (W & 1) == 0, "H and W must be even for 2x2 s2 pooling");

        // vectorize over C
        TORCH_CHECK((C & 3) == 0, "C must be divisible by 4 for NHWC vec4 fast-path");
        int C4 = C >> 2;

        auto out = torch::empty({N, C, H/2, W/2}, x.options().memory_format(at::MemoryFormat::ChannelsLast));

        const float* in_f = x.data_ptr<float>();
        float* out_f = out.data_ptr<float>();

        uintptr_t in_addr = reinterpret_cast<uintptr_t>(in_f);
        uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_f);
        TORCH_CHECK((in_addr & 0xF) == 0 && (out_addr & 0xF) == 0, "NHWC vec4 requires 16B alignment");

        int OH = H >> 1;
        int OW = W >> 1;
        int64_t total = (int64_t)N * OH * OW * C4;

        constexpr int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = blocks > 131072 ? 131072 : blocks;
        blocks = blocks < 1 ? 1 : blocks;

        relu_maxpool2x2_s2_nhwc_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(in_f),
            reinterpret_cast<float4*>(out_f),
            N, H, W, C4
        );
        return out;
    }

    // NCHW contiguous fallback
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous in either contiguous (NCHW) or channels_last (NHWC) format");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    TORCH_CHECK((H & 1) == 0 && (W & 1) == 0, "H and W must be even for 2x2 s2 pooling");

    auto out = torch::empty({N, C, H/2, W/2}, x.options());

    int OH = H >> 1;
    int OW = W >> 1;
    int64_t total = (int64_t)N * C * OH * OW;

    constexpr int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    blocks = blocks > 131072 ? 131072 : blocks;
    blocks = blocks < 1 ? 1 : blocks;

    relu_maxpool2x2_s2_nchw_scalar<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );
    return out;
}
"""

cpp_src = r"""
torch::Tensor relu_inplace_cuda(torch::Tensor x);
torch::Tensor relu_maxpool2x2_s2_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_vgg16_fused_relu_pool_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_inplace_cuda", "relu_maxpool2x2_s2_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Convolution blocks
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Classifier
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.drop1 = nn.Dropout(p=0.0)
        self.drop2 = nn.Dropout(p=0.0)

    def _relu_inplace(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous():
            return torch.relu_(x)
        return custom_ops_lib.relu_inplace_cuda(x)

    def _relu_pool(self, x: torch.Tensor) -> torch.Tensor:
        # Fused relu+pool requires contiguous either NCHW or channels_last.
        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4):
            return torch.nn.functional.max_pool2d(torch.relu(x), kernel_size=2, stride=2)

        if x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last):
            # NHWC vec4 requires C%4==0 and even H/W; otherwise fall back.
            H, W = x.shape[2], x.shape[3]
            if (H % 2 == 0) and (W % 2 == 0):
                if x.is_contiguous(memory_format=torch.channels_last):
                    if (x.shape[1] % 4) == 0:
                        return custom_ops_lib.relu_maxpool2x2_s2_cuda(x)
                    else:
                        return torch.nn.functional.max_pool2d(torch.relu(x), kernel_size=2, stride=2)
                else:
                    return custom_ops_lib.relu_maxpool2x2_s2_cuda(x)

        return torch.nn.functional.max_pool2d(torch.relu(x), kernel_size=2, stride=2)

    def forward(self, x):
        # Prefer channels_last to (a) enable fused NHWC vec4 path and
        # (b) reduce likelihood of NCHW<->NHWC transforms in cuDNN pipelines.
        if x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous(memory_format=torch.channels_last)

        # Block 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self._relu_pool(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self._relu_pool(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self._relu_pool(x)

        # Block 4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self._relu_pool(x)

        # Block 5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self._relu_pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self._relu_inplace(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self._relu_inplace(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x