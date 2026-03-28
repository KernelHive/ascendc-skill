import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension: fused BatchNorm(eval) + Softmax over W for NHWC (channels_last) float32
# Input/output: NHWC contiguous (i.e., torch.channels_last 4D tensor)
# Softmax dimension: W (width), matching NCHW softmax(dim=-1)
# -------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static __device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}
static __device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static __device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    // smem size: num_warps floats
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float out = -INFINITY;
    if (warp == 0) {
        float x = (lane < (blockDim.x + 31) / 32) ? smem[lane] : -INFINITY;
        x = warp_reduce_max(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    out = smem[0];
    return out;
}

static __device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        float x = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    out = smem[0];
    return out;
}

// Fused BN(eval) + softmax along W for NHWC.
// Grid: one block per (n,h,c). Softmax over W.
// Memory layout NHWC contiguous => W stride = C; C is last-but-one.
__global__ __launch_bounds__(256, 2)
void bn_softmax_w_nhwc_f32_kernel(
    const float* __restrict__ x,  // NHWC
    float* __restrict__ y,        // NHWC
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    int N, int H, int W, int C,
    float eps
) {
    const int tid = (int)threadIdx.x;
    const int nhw = (int)blockIdx.x; // 0..N*H*C-1
    const int c = nhw % C;
    const int nh = nhw / C;
    const int h = nh % H;
    const int n = nh / H;

    // Base offset for (n,h,0,c): index = ((n*H + h)*W + w)*C + c
    const int base = ((n * H + h) * W) * C + c;

    // Load BN params through read-only cache (compiler may use LDG)
    const float g = gamma ? gamma[c] : 1.0f;
    const float b = beta ? beta[c] : 0.0f;
    const float m = mean ? mean[c] : 0.0f;
    const float v = var ? var[c] : 1.0f;
    const float inv_std = rsqrtf(v + eps);

    // We want to vectorize across W when C%4==0 and base pointer is 16B aligned.
    // Because NHWC contiguous and base advances by C each w, we can use float4
    // only if C is multiple of 4 and c is aligned to 4 lanes.
    const bool vec_ok = ((C & 3) == 0) && ((c & 3) == 0);

    // small shared memory for warp leaders
    extern __shared__ float smem[];
    float* sreduce = smem;

    float row_max = -INFINITY;

    if (vec_ok) {
        // Process four channels at once; this threadblock is for c and c+0..3 (but grid is per-c).
        // We only allow vec_ok when c%4==0; for c+1..3 blocks, vec_ok is false, so no aliasing.
        // Therefore we only use float4 in blocks where c%4==0, and those blocks compute and write
        // only the first lane (c) to keep semantics unchanged. Still helps coalescing by reading 16B.
        const float4* x4 = reinterpret_cast<const float4*>(x + base);
        // Cache BN-transformed values for this thread's W positions (streaming).
        // For W up to 512 and threads up to 256, each thread handles <=4 positions: keep in registers.
        float z_cache[8];
        int w_cache[8];
        int count = 0;

        for (int w = tid; w < W; w += (int)blockDim.x) {
            // x4 index corresponds to element ((..., w)*C + c) / 4
            const int idx4 = (w * C) >> 2;
            float4 v4 = x4[idx4];
            float z = (v4.x - m) * inv_std * g + b; // lane x corresponds to channel c
            z_cache[count] = z;
            w_cache[count] = w;
            count++;
            row_max = fmaxf(row_max, z);
        }
        row_max = block_reduce_max(row_max, sreduce);

        float row_sum = 0.0f;
        for (int i = 0; i < count; i++) {
            row_sum += expf(z_cache[i] - row_max);
        }
        row_sum = block_reduce_sum(row_sum, sreduce);
        float inv = 1.0f / row_sum;

        for (int i = 0; i < count; i++) {
            int w = w_cache[i];
            y[base + w * C] = expf(z_cache[i] - row_max) * inv;
        }
    } else {
        // Scalar path: cache z for owned W positions to avoid recomputing BN.
        float z_cache[8];
        int w_cache[8];
        int count = 0;

        for (int w = tid; w < W; w += (int)blockDim.x) {
            float xv = x[base + w * C];
            float z = (xv - m) * inv_std * g + b;
            z_cache[count] = z;
            w_cache[count] = w;
            count++;
            row_max = fmaxf(row_max, z);
        }
        row_max = block_reduce_max(row_max, sreduce);

        float row_sum = 0.0f;
        for (int i = 0; i < count; i++) {
            row_sum += expf(z_cache[i] - row_max);
        }
        row_sum = block_reduce_sum(row_sum, sreduce);
        float inv = 1.0f / row_sum;

        for (int i = 0; i < count; i++) {
            int w = w_cache[i];
            y[base + w * C] = expf(z_cache[i] - row_max) * inv;
        }
    }
}

torch::Tensor bn_softmax_w_nhwc_f32_cuda(
    torch::Tensor x,              // NHWC contiguous (channels_last)
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 4, "expected 4D NHWC tensor");

    const int64_t N = x.size(0);
    const int64_t H = x.size(1);
    const int64_t W = x.size(2);
    const int64_t C = x.size(3);

    TORCH_CHECK(gamma.defined() && beta.defined() && running_mean.defined() && running_var.defined(),
                "BN params must be defined");
    CHECK_CUDA(gamma); CHECK_CUDA(beta); CHECK_CUDA(running_mean); CHECK_CUDA(running_var);
    CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta); CHECK_CONTIGUOUS(running_mean); CHECK_CONTIGUOUS(running_var);
    CHECK_FLOAT(gamma); CHECK_FLOAT(beta); CHECK_FLOAT(running_mean); CHECK_FLOAT(running_var);
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C && running_mean.numel() == C && running_var.numel() == C,
                "BN params must have shape [C] matching input");

    auto y = torch::empty_like(x);

    // Threads: tune for W. For W<=256, 128 threads often enough; for W up to 512 use 256.
    int threads = (W <= 256) ? 128 : 256;
    int blocks = (int)(N * H * C);
    int num_warps = (threads + 31) / 32;
    size_t shmem = (size_t)num_warps * sizeof(float);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bn_softmax_w_nhwc_f32_kernel<<<blocks, threads, shmem, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (const float*)running_mean.data_ptr<float>(),
        (const float*)running_var.data_ptr<float>(),
        (int)N, (int)H, (int)W, (int)C,
        (float)eps
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor bn_softmax_w_nhwc_f32_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_unet_bn_softmax_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bn_softmax_w_nhwc_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model: keep channels_last through eval forward to avoid repeated conversions
# -------------------------

class DoubleConvNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def _bn_softmax_eval_channels_last(self, x_nhwc: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
        # x_nhwc: NHWC float32 contiguous, CUDA, eval only.
        return custom_ops_lib.bn_softmax_w_nhwc_f32_cuda(
            x_nhwc,
            bn.weight.contiguous(),
            bn.bias.contiguous(),
            bn.running_mean.contiguous(),
            bn.running_var.contiguous(),
            float(bn.eps),
        )

    def forward(self, x):
        # x is NCHW or channels_last; we preserve memory_format if possible.
        x = self.conv1(x)
        if x.is_cuda and (not self.training) and x.dtype == torch.float32:
            # Ensure channels_last once here; downstream ops in ModelNew keep it.
            x = x.contiguous(memory_format=torch.channels_last)
            # Convert to NHWC view (no permute needed for channels_last contiguous: still NCHW logical)
            # But our kernel expects explicit NHWC indexing, so we must permute to NHWC.
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            y_nhwc = self._bn_softmax_eval_channels_last(x_nhwc, self.bn1)
            x = y_nhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        else:
            x = self.bn1(x)
            x = torch.softmax(x, dim=-1)

        x = self.conv2(x)
        if x.is_cuda and (not self.training) and x.dtype == torch.float32:
            x = x.contiguous(memory_format=torch.channels_last)
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            y_nhwc = self._bn_softmax_eval_channels_last(x_nhwc, self.bn2)
            x = y_nhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        else:
            x = self.bn2(x)
            x = torch.softmax(x, dim=-1)

        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        self.encoder1 = DoubleConvNew(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConvNew(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConvNew(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConvNew(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvNew(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvNew(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvNew(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvNew(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvNew(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # In eval+CUDA, keep channels_last for the whole model to reduce conversion churn.
        if x.is_cuda and (not self.training) and x.dtype == torch.float32:
            x = x.contiguous(memory_format=torch.channels_last)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1).contiguous(memory_format=dec4.stride() == dec4.contiguous().stride() and torch.contiguous_format or torch.channels_last)
        # safer: preserve channels_last if CUDA+eval, else contiguous
        if dec4.is_cuda and (not self.training) and dec4.dtype == torch.float32:
            dec4 = dec4.contiguous(memory_format=torch.channels_last)
        else:
            dec4 = dec4.contiguous()
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        if dec3.is_cuda and (not self.training) and dec3.dtype == torch.float32:
            dec3 = dec3.contiguous(memory_format=torch.channels_last)
        else:
            dec3 = dec3.contiguous()
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        if dec2.is_cuda and (not self.training) and dec2.dtype == torch.float32:
            dec2 = dec2.contiguous(memory_format=torch.channels_last)
        else:
            dec2 = dec2.contiguous()
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        if dec1.is_cuda and (not self.training) and dec1.dtype == torch.float32:
            dec1 = dec1.contiguous(memory_format=torch.channels_last)
        else:
            dec1 = dec1.contiguous()
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)
        return out