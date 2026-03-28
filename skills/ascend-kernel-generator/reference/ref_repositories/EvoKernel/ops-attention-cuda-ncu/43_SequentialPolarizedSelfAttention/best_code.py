import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA extension for Sequential Polarized Self-Attention
#
# Key improvements vs baseline:
# - Channel stage: compute softmax(channel_wq) once per batch and reuse via shared memory
#   across all C2 channels (warp-per-channel inside a block-per-batch kernel).
# - Spatial stage: warp-per-(b,hw) softmax(C2)+dot, using warp reductions and __ldg.
# - Fuse spatial sigmoid + multiply into one kernel (avoid extra global traffic).
# - Grid-stride loops for elementwise channel-weight application.
# -----------------------------------------------------------------------------

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

static inline __device__ float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

static inline __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static inline __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

// ------------------------------------
// Channel stage: block-per-batch
// - blockDim.x must be multiple of 32
// - shared memory holds p[HW]
// Mapping:
// - Compute softmax over HW using all threads in block (grid-stride over HW).
// - Then each warp handles one c2 channel: warp_id = tid/32; c2 = warp_id + warp_base
//   and loops over c2 in steps of warps_per_block.
__global__ void channel_wz_block_per_batch(
    const float* __restrict__ wv,   // [B, C2, HW] contiguous
    const float* __restrict__ wq,   // [B, HW] contiguous view of [B,1,H,W]
    float* __restrict__ wz,         // [B, C2]
    int C2, int HW)
{
    int b = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = (int)(blockDim.x >> 5);

    const float* wq_ptr = wq + b * HW;

    extern __shared__ float smem[];
    float* s_p = smem; // [HW]

    // 1) max over HW (block reduction via warp partials)
    float mx_local = -INFINITY;
    for (int i = tid; i < HW; i += (int)blockDim.x) {
        mx_local = fmaxf(mx_local, LDG(wq_ptr + i));
    }
    float mx_warp = warp_reduce_max(mx_local);

    __shared__ float s_warp_max[8]; // supports up to 8 warps (blockDim.x<=256)
    if (lane == 0) s_warp_max[warp] = mx_warp;
    __syncthreads();

    float mx = -INFINITY;
    if (warp == 0) {
        float v = (lane < warps_per_block) ? s_warp_max[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) s_warp_max[0] = v; // reuse slot 0 to broadcast
    }
    __syncthreads();
    mx = s_warp_max[0];

    // 2) denom and fill p
    float sum_local = 0.f;
    for (int i = tid; i < HW; i += (int)blockDim.x) {
        sum_local += expf(LDG(wq_ptr + i) - mx);
    }
    float sum_warp = warp_reduce_sum(sum_local);

    __shared__ float s_warp_sum[8];
    if (lane == 0) s_warp_sum[warp] = sum_warp;
    __syncthreads();

    float denom = 0.f;
    if (warp == 0) {
        float v = (lane < warps_per_block) ? s_warp_sum[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp_sum[0] = v; // broadcast denom
    }
    __syncthreads();
    denom = s_warp_sum[0];
    float inv = (denom > 0.f) ? (1.f / denom) : 1.f;

    for (int i = tid; i < HW; i += (int)blockDim.x) {
        s_p[i] = expf(LDG(wq_ptr + i) - mx) * inv;
    }
    __syncthreads();

    // 3) dot for channels: warp-per-channel
    // Each warp accumulates for one c2 at a time, striding over channels.
    for (int c2 = warp; c2 < C2; c2 += warps_per_block) {
        const float* wv_ptr = wv + ((b * C2 + c2) * HW);
        float acc = 0.f;
        for (int i = lane; i < HW; i += 32) {
            acc += LDG(wv_ptr + i) * s_p[i];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) wz[b * C2 + c2] = acc;
    }
}

// ------------------------------------
// Apply channel weight: out = x * w[b,c]
// Use grid-stride to keep occupancy; read-only cache for w.
__global__ void apply_channel_weight_gs(
    const float* __restrict__ x,   // [B,C,HW]
    const float* __restrict__ w,   // [B,C] stored as [B,C,1,1] contiguous => first B*C floats
    float* __restrict__ out,
    int C, int HW, int N)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < N; i += stride) {
        int t = i / HW;
        int c = t % C;
        int b = t / C;
        out[i] = x[i] * LDG(w + (b * C + c));
    }
}

// ------------------------------------
// Spatial stage: warp-per-(b,hw) softmax(C2) + dot
__global__ void spatial_wz_warp(
    const float* __restrict__ wv,       // [B,C2,HW]
    const float* __restrict__ wq_avg,   // [B,C2]
    float* __restrict__ wz_hw,          // [B,HW]
    int B, int C2, int HW)
{
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = (int)(blockDim.x >> 5);

    int linear = (int)blockIdx.x * warps_per_block + warp; // over (b,hw)
    int total = B * HW;
    if (linear >= total) return;

    int b = linear / HW;
    int hw = linear - b * HW;

    const float* q = wq_avg + b * C2;

    float mx = -INFINITY;
    for (int c2 = lane; c2 < C2; c2 += 32) {
        mx = fmaxf(mx, LDG(q + c2));
    }
    mx = warp_reduce_max(mx);

    float denom = 0.f;
    for (int c2 = lane; c2 < C2; c2 += 32) {
        denom += expf(LDG(q + c2) - mx);
    }
    denom = warp_reduce_sum(denom);
    float inv = (denom > 0.f) ? (1.f / denom) : 1.f;

    float acc = 0.f;
    for (int c2 = lane; c2 < C2; c2 += 32) {
        float p = expf(LDG(q + c2) - mx) * inv;
        const float* wv_ptr = wv + ((b * C2 + c2) * HW);
        acc += p * LDG(wv_ptr + hw);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) wz_hw[b * HW + hw] = acc;
}

// ------------------------------------
// Fused spatial apply: out = channel_out * sigmoid(wz_hw[b,hw])
__global__ void apply_spatial_weight_fused_gs(
    const float* __restrict__ x,      // [B,C,HW]
    const float* __restrict__ wz_hw,  // [B,HW]
    float* __restrict__ out,
    int C, int HW, int N)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < N; i += stride) {
        int hw = i % HW;
        int t = i / HW;
        int b = t / C;
        float w = sigmoidf(LDG(wz_hw + (b * HW + hw)));
        out[i] = x[i] * w;
    }
}

// ------------------------------------
// Bindings
torch::Tensor channel_wz_cuda(torch::Tensor channel_wv, torch::Tensor channel_wq) {
    TORCH_CHECK(channel_wv.is_cuda() && channel_wq.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(channel_wv.scalar_type() == at::ScalarType::Float, "channel_wv must be float32");
    TORCH_CHECK(channel_wq.scalar_type() == at::ScalarType::Float, "channel_wq must be float32");
    TORCH_CHECK(channel_wv.is_contiguous(), "channel_wv must be contiguous");
    TORCH_CHECK(channel_wq.is_contiguous(), "channel_wq must be contiguous");
    TORCH_CHECK(channel_wv.dim() == 4, "channel_wv must be [B,C2,H,W]");
    TORCH_CHECK(channel_wq.dim() == 4, "channel_wq must be [B,1,H,W]");

    int B = (int)channel_wv.size(0);
    int C2 = (int)channel_wv.size(1);
    int H = (int)channel_wv.size(2);
    int W = (int)channel_wv.size(3);
    int HW = H * W;

    TORCH_CHECK((int)channel_wq.size(0) == B && (int)channel_wq.size(1) == 1 &&
                (int)channel_wq.size(2) == H && (int)channel_wq.size(3) == W,
                "channel_wq shape mismatch");

    auto wz = torch::empty({B, C2}, channel_wv.options());

    // block-per-batch; choose 256 threads (8 warps) for good occupancy
    int threads = 256;
    int blocks = B;
    size_t shmem = (size_t)HW * sizeof(float);

    channel_wz_block_per_batch<<<blocks, threads, shmem>>>(
        (const float*)channel_wv.data_ptr<float>(),
        (const float*)channel_wq.data_ptr<float>(), // treated as [B,HW] due to contiguous [B,1,H,W]
        (float*)wz.data_ptr<float>(),
        C2, HW
    );
    return wz;
}

torch::Tensor apply_channel_weight_cuda(torch::Tensor x, torch::Tensor channel_weight) {
    TORCH_CHECK(x.is_cuda() && channel_weight.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(channel_weight.scalar_type() == at::ScalarType::Float, "channel_weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(channel_weight.is_contiguous(), "channel_weight must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");
    TORCH_CHECK(channel_weight.dim() == 4, "channel_weight must be [B,C,1,1]");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;
    int N = B * C * HW;

    TORCH_CHECK((int)channel_weight.size(0) == B && (int)channel_weight.size(1) == C &&
                (int)channel_weight.size(2) == 1 && (int)channel_weight.size(3) == 1,
                "channel_weight shape mismatch");

    auto out = torch::empty_like(x);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = blocks > 8192 ? 8192 : blocks;

    apply_channel_weight_gs<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)channel_weight.data_ptr<float>(), // first B*C floats correspond to [B,C,1,1]
        (float*)out.data_ptr<float>(),
        C, HW, N
    );
    return out;
}

torch::Tensor spatial_wz_cuda(torch::Tensor spatial_wv, torch::Tensor spatial_wq_avg) {
    TORCH_CHECK(spatial_wv.is_cuda() && spatial_wq_avg.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(spatial_wv.scalar_type() == at::ScalarType::Float, "spatial_wv must be float32");
    TORCH_CHECK(spatial_wq_avg.scalar_type() == at::ScalarType::Float, "spatial_wq_avg must be float32");
    TORCH_CHECK(spatial_wv.is_contiguous(), "spatial_wv must be contiguous");
    TORCH_CHECK(spatial_wq_avg.is_contiguous(), "spatial_wq_avg must be contiguous");
    TORCH_CHECK(spatial_wv.dim() == 4, "spatial_wv must be [B,C2,H,W]");
    TORCH_CHECK(spatial_wq_avg.dim() == 2, "spatial_wq_avg must be [B,C2]");

    int B = (int)spatial_wv.size(0);
    int C2 = (int)spatial_wv.size(1);
    int H = (int)spatial_wv.size(2);
    int W = (int)spatial_wv.size(3);
    int HW = H * W;

    TORCH_CHECK((int)spatial_wq_avg.size(0) == B && (int)spatial_wq_avg.size(1) == C2,
                "spatial_wq_avg shape mismatch");

    auto wz_hw = torch::empty({B, HW}, spatial_wv.options());

    // 128 threads => 4 warps per block
    int threads = 128;
    int warps_per_block = threads / 32;
    int total = B * HW;
    int blocks = (total + warps_per_block - 1) / warps_per_block;

    spatial_wz_warp<<<blocks, threads>>>(
        (const float*)spatial_wv.data_ptr<float>(),
        (const float*)spatial_wq_avg.data_ptr<float>(),
        (float*)wz_hw.data_ptr<float>(),
        B, C2, HW
    );
    return wz_hw;
}

torch::Tensor apply_spatial_weight_cuda(torch::Tensor channel_out, torch::Tensor wz_hw) {
    TORCH_CHECK(channel_out.is_cuda() && wz_hw.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(channel_out.scalar_type() == at::ScalarType::Float, "channel_out must be float32");
    TORCH_CHECK(wz_hw.scalar_type() == at::ScalarType::Float, "wz_hw must be float32");
    TORCH_CHECK(channel_out.is_contiguous(), "channel_out must be contiguous");
    TORCH_CHECK(wz_hw.is_contiguous(), "wz_hw must be contiguous");
    TORCH_CHECK(channel_out.dim() == 4, "channel_out must be [B,C,H,W]");
    TORCH_CHECK(wz_hw.dim() == 2, "wz_hw must be [B,HW]");

    int B = (int)channel_out.size(0);
    int C = (int)channel_out.size(1);
    int H = (int)channel_out.size(2);
    int W = (int)channel_out.size(3);
    int HW = H * W;
    int N = B * C * HW;

    TORCH_CHECK((int)wz_hw.size(0) == B && (int)wz_hw.size(1) == HW, "wz_hw shape mismatch");

    auto out = torch::empty_like(channel_out);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = blocks > 8192 ? 8192 : blocks;

    apply_spatial_weight_fused_gs<<<blocks, threads>>>(
        (const float*)channel_out.data_ptr<float>(),
        (const float*)wz_hw.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        C, HW, N
    );
    return out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>

torch::Tensor channel_wz_cuda(torch::Tensor channel_wv, torch::Tensor channel_wq);
torch::Tensor apply_channel_weight_cuda(torch::Tensor x, torch::Tensor channel_weight);
torch::Tensor spatial_wz_cuda(torch::Tensor spatial_wv, torch::Tensor spatial_wq_avg);
torch::Tensor apply_spatial_weight_cuda(torch::Tensor channel_out, torch::Tensor wz_hw);
"""

custom_ops_lib = load_inline(
    name="custom_sequential_polarized_self_attention_ops_opt3",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=[
        "channel_wz_cuda",
        "apply_channel_weight_cuda",
        "spatial_wz_cuda",
        "apply_spatial_weight_cuda",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Sequential Polarized Self-Attention with optimized custom CUDA kernels.
    Custom CUDA path supports CUDA float32 contiguous tensors; otherwise falls back to PyTorch.
    """

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=1)
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()

        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        self.custom_ops_lib = custom_ops_lib
        self.channel = channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32):
            b, c, h, w = x.size()
            channel_wv = self.ch_wv(x)
            channel_wq = self.ch_wq(x)
            channel_wv = channel_wv.reshape(b, c // 2, -1)
            channel_wq = channel_wq.reshape(b, -1, 1)
            channel_wq = torch.softmax(channel_wq, dim=1)
            channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
            channel_weight = self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            ).permute(0, 2, 1).reshape(b, c, 1, 1)
            channel_out = channel_weight * x

            spatial_wv = self.sp_wv(channel_out)
            spatial_wq = self.sp_wq(channel_out)
            spatial_wq = self.agp(spatial_wq)
            spatial_wv = spatial_wv.reshape(b, c // 2, -1)
            spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
            spatial_wq = torch.softmax(spatial_wq, dim=-1)
            spatial_wz = torch.matmul(spatial_wq, spatial_wv)
            spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
            return spatial_weight * channel_out

        if not x.is_contiguous():
            x = x.contiguous()

        b, c, h, w = x.shape
        c2 = c // 2

        # Channel attention: wz_vec [B,C2] computed with softmax(wq) once per batch
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        if not channel_wv.is_contiguous():
            channel_wv = channel_wv.contiguous()
        if not channel_wq.is_contiguous():
            channel_wq = channel_wq.contiguous()

        channel_wz_vec = self.custom_ops_lib.channel_wz_cuda(channel_wv, channel_wq)  # [B,C2]
        channel_wz = channel_wz_vec.view(b, c2, 1, 1)

        # channel_weight via PyTorch ops (small tensors, good kernels)
        cw = self.ch_wz(channel_wz)  # [B,C,1,1]
        cw_ln_in = cw.reshape(b, c, 1).permute(0, 2, 1)  # [B,1,C]
        channel_weight = self.sigmoid(self.ln(cw_ln_in)).permute(0, 2, 1).reshape(b, c, 1, 1).contiguous()

        channel_out = self.custom_ops_lib.apply_channel_weight_cuda(x, channel_weight)

        # Spatial attention: compute wz_hw [B,HW] then fused sigmoid+mul
        spatial_wv = self.sp_wv(channel_out)
        spatial_wq = self.sp_wq(channel_out)
        if not spatial_wv.is_contiguous():
            spatial_wv = spatial_wv.contiguous()
        if not spatial_wq.is_contiguous():
            spatial_wq = spatial_wq.contiguous()

        spatial_wq_avg = self.agp(spatial_wq).view(b, c2).contiguous()  # [B,C2]
        wz_hw = self.custom_ops_lib.spatial_wz_cuda(spatial_wv, spatial_wq_avg)      # [B,HW]
        out = self.custom_ops_lib.apply_spatial_weight_cuda(channel_out, wz_hw)     # [B,C,H,W]
        return out