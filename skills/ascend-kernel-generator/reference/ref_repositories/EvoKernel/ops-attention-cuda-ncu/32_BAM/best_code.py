import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ---------------------------
# Original submodules (kept)
# ---------------------------

class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.mlp(y)
        y = self.bn(y).view(b, c, 1, 1)
        return y.expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3, dilation_val=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(y)
        return y.expand_as(x)


# ---------------------------
# Custom CUDA ops
# 1) Existing fused tail (3 expanded tensors): out = x + x * sigmoid(channel + spatial)
# 2) New fused tail from compact tensors:
#    channel_bc: (B,C)  spatial_bhw: (B,H,W) or (B,HW)
#    out[b,c,h,w] = x + x * sigmoid(channel_bc[b,c] + spatial_bhw[b,h,w])
# ---------------------------

bam_fused_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float ro_load_f32(const float* __restrict__ p) { return __ldg(p); }
#else
__device__ __forceinline__ float ro_load_f32(const float* __restrict__ p) { return *p; }
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float4 load_f4(const float4* __restrict__ p) { return *p; }
__device__ __forceinline__ void store_f4(float4* __restrict__ p, const float4& v) { *p = v; }

static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// -------------------------------------
// Kernel A: 3-input expanded tensors
// -------------------------------------

__global__ __launch_bounds__(128, 6)
void bam_fused_vec4_kernel_128(
    const float* __restrict__ x,
    const float* __restrict__ channel,
    const float* __restrict__ spatial,
    float* __restrict__ out,
    int64_t n4
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    const float4* __restrict__ c4 = reinterpret_cast<const float4*>(channel);
    const float4* __restrict__ s4 = reinterpret_cast<const float4*>(spatial);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

    for (int64_t i = tid; i < n4; i += stride) {
        // use __ldg on scalar path only; for float4 we rely on L1/L2
        const float4 xv = load_f4(x4 + i);
        const float4 cv = load_f4(c4 + i);
        const float4 sv = load_f4(s4 + i);

        float4 ov;
        float a0 = cv.x + sv.x; float g0 = sigmoidf_fast(a0); ov.x = fmaf(xv.x, g0, xv.x);
        float a1 = cv.y + sv.y; float g1 = sigmoidf_fast(a1); ov.y = fmaf(xv.y, g1, xv.y);
        float a2 = cv.z + sv.z; float g2 = sigmoidf_fast(a2); ov.z = fmaf(xv.z, g2, xv.z);
        float a3 = cv.w + sv.w; float g3 = sigmoidf_fast(a3); ov.w = fmaf(xv.w, g3, xv.w);

        store_f4(o4 + i, ov);
    }
}

__global__ __launch_bounds__(128, 6)
void bam_fused_scalar_kernel_128(
    const float* __restrict__ x,
    const float* __restrict__ channel,
    const float* __restrict__ spatial,
    float* __restrict__ out,
    int64_t n
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float xv = ro_load_f32(x + i);
        float a  = ro_load_f32(channel + i) + ro_load_f32(spatial + i);
        float g  = sigmoidf_fast(a);
        out[i] = fmaf(xv, g, xv);
    }
}

// -------------------------------------
// Kernel B: compact channel/spatial
// channel_bc: (B,C)
// spatial_bhw: (B,HW) contiguous
// x/out: (B,C,H,W) contiguous NCHW
//
// Use vectorized float4 along the W dimension when possible.
// Specialized fast-path for HW==49 (7x7): avoid div/mod by arbitrary HW.
// -------------------------------------

__global__ __launch_bounds__(128, 6)
void bam_fused_compact_hw49_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ channel_bc,  // B*C
    const float* __restrict__ spatial_bhw, // B*49
    float* __restrict__ out,
    int B, int C
) {
    // Total elements = B*C*49; process float4 => groups of 4 contiguous elements in the last dimension.
    // Layout NCHW contiguous: index = (((b*C + c)*49) + hw)
    const int64_t n = (int64_t)B * (int64_t)C * 49LL;
    const int64_t n4 = n >> 2;

    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

    // Each float4 corresponds to 4 consecutive hw positions within the same (b,c) most of the time.
    // For HW=49, boundaries can split; handle by computing base scalar index for this float4.
    for (int64_t i4 = tid; i4 < n4; i4 += stride) {
        const int64_t base = i4 << 2; // scalar element index
        // base = ((b*C + c)*49 + hw)
        const int64_t bc = base / 49LL;
        const int64_t hw = base - bc * 49LL;
        const int b = (int)(bc / (int64_t)C);
        const int c = (int)(bc - (int64_t)b * (int64_t)C);

        // Load channel scalar once per float4
        const float ch = ro_load_f32(channel_bc + (int64_t)b * (int64_t)C + (int64_t)c);

        // Load x float4
        const float4 xv = load_f4(x4 + i4);

        // Spatial loads: need 4 scalar loads (hw..hw+3). Usually within [0,48], but may cross 49 boundary.
        // If crossing boundary, we fall back to scalar compute for those lanes by recomputing indices.
        float4 ov;

        // lane 0
        {
            const int64_t idx0 = base;
            const int64_t bc0 = idx0 / 49LL;
            const int64_t hw0 = idx0 - bc0 * 49LL;
            const int b0 = (int)(bc0 / (int64_t)C);
            const float sp0 = ro_load_f32(spatial_bhw + (int64_t)b0 * 49LL + hw0);
            const float g0 = sigmoidf_fast(ch + sp0);
            ov.x = fmaf(xv.x, g0, xv.x);
        }
        // lane 1
        {
            const int64_t idx1 = base + 1;
            const int64_t bc1 = idx1 / 49LL;
            const int64_t hw1 = idx1 - bc1 * 49LL;
            const int b1 = (int)(bc1 / (int64_t)C);
            // ch is only valid if bc1==bc; if boundary crossed, reload correct channel
            const int64_t bc1_int = bc1;
            const float ch1 = (bc1_int == bc) ? ch : ro_load_f32(channel_bc + (int64_t)b1 * (int64_t)C + (int64_t)((bc1_int - (int64_t)b1*(int64_t)C)));
            const float sp1 = ro_load_f32(spatial_bhw + (int64_t)b1 * 49LL + hw1);
            const float g1 = sigmoidf_fast(ch1 + sp1);
            ov.y = fmaf(xv.y, g1, xv.y);
        }
        // lane 2
        {
            const int64_t idx2 = base + 2;
            const int64_t bc2 = idx2 / 49LL;
            const int64_t hw2 = idx2 - bc2 * 49LL;
            const int b2 = (int)(bc2 / (int64_t)C);
            const int64_t bc2_int = bc2;
            const float ch2 = (bc2_int == bc) ? ch : ro_load_f32(channel_bc + (int64_t)b2 * (int64_t)C + (int64_t)((bc2_int - (int64_t)b2*(int64_t)C)));
            const float sp2 = ro_load_f32(spatial_bhw + (int64_t)b2 * 49LL + hw2);
            const float g2 = sigmoidf_fast(ch2 + sp2);
            ov.z = fmaf(xv.z, g2, xv.z);
        }
        // lane 3
        {
            const int64_t idx3 = base + 3;
            const int64_t bc3 = idx3 / 49LL;
            const int64_t hw3 = idx3 - bc3 * 49LL;
            const int b3 = (int)(bc3 / (int64_t)C);
            const int64_t bc3_int = bc3;
            const float ch3 = (bc3_int == bc) ? ch : ro_load_f32(channel_bc + (int64_t)b3 * (int64_t)C + (int64_t)((bc3_int - (int64_t)b3*(int64_t)C)));
            const float sp3 = ro_load_f32(spatial_bhw + (int64_t)b3 * 49LL + hw3);
            const float g3 = sigmoidf_fast(ch3 + sp3);
            ov.w = fmaf(xv.w, g3, xv.w);
        }

        store_f4(o4 + i4, ov);
    }
}

__global__ __launch_bounds__(128, 6)
void bam_fused_compact_generic_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ channel_bc,  // B*C
    const float* __restrict__ spatial_bhw, // B*HW
    float* __restrict__ out,
    int B, int C, int HW
) {
    const int64_t n = (int64_t)B * (int64_t)C * (int64_t)HW;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < n; idx += stride) {
        const int64_t bc = idx / (int64_t)HW;
        const int64_t hw = idx - bc * (int64_t)HW;
        const int b = (int)(bc / (int64_t)C);
        const int c = (int)(bc - (int64_t)b * (int64_t)C);

        const float xv = ro_load_f32(x + idx);
        const float ch = ro_load_f32(channel_bc + (int64_t)b * (int64_t)C + (int64_t)c);
        const float sp = ro_load_f32(spatial_bhw + (int64_t)b * (int64_t)HW + hw);
        const float g  = sigmoidf_fast(ch + sp);
        out[idx] = fmaf(xv, g, xv);
    }
}

// -------------------------------------
// C++ interfaces
// -------------------------------------

torch::Tensor bam_fused_cuda(torch::Tensor x, torch::Tensor channel, torch::Tensor spatial) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(channel.is_cuda(), "channel must be a CUDA tensor");
    TORCH_CHECK(spatial.is_cuda(), "spatial must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(channel.dtype() == torch::kFloat32, "channel must be float32");
    TORCH_CHECK(spatial.dtype() == torch::kFloat32, "spatial must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(channel.is_contiguous(), "channel must be contiguous");
    TORCH_CHECK(spatial.is_contiguous(), "spatial must be contiguous");

    TORCH_CHECK(x.sizes() == channel.sizes(), "x and channel must have same shape");
    TORCH_CHECK(x.sizes() == spatial.sizes(), "x and spatial must have same shape");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);

    const int64_t n = x.numel();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const uintptr_t x_ptr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t c_ptr = (uintptr_t)channel.data_ptr<float>();
    const uintptr_t s_ptr = (uintptr_t)spatial.data_ptr<float>();
    const uintptr_t o_ptr = (uintptr_t)out.data_ptr<float>();

    const bool aligned16 = ((x_ptr | c_ptr | s_ptr | o_ptr) & 0xF) == 0;
    const bool n_div4 = (n & 3LL) == 0;

    const int threads = 128;
    const int64_t max_blocks = 8192;
    int64_t blocks;

    if (aligned16 && n_div4) {
        const int64_t n4 = n >> 2;
        blocks = ceil_div_i64(n4, (int64_t)threads);
        if (blocks < 1) blocks = 1;
        if (blocks > max_blocks) blocks = max_blocks;

        bam_fused_vec4_kernel_128<<<(unsigned int)blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            channel.data_ptr<float>(),
            spatial.data_ptr<float>(),
            out.data_ptr<float>(),
            n4
        );
    } else {
        blocks = ceil_div_i64(n, (int64_t)threads);
        if (blocks < 1) blocks = 1;
        if (blocks > max_blocks) blocks = max_blocks;

        bam_fused_scalar_kernel_128<<<(unsigned int)blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            channel.data_ptr<float>(),
            spatial.data_ptr<float>(),
            out.data_ptr<float>(),
            n
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor bam_fused_compact_cuda(torch::Tensor x, torch::Tensor channel_bc, torch::Tensor spatial_bhw) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(channel_bc.is_cuda(), "channel_bc must be a CUDA tensor");
    TORCH_CHECK(spatial_bhw.is_cuda(), "spatial_bhw must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(channel_bc.dtype() == torch::kFloat32, "channel_bc must be float32");
    TORCH_CHECK(spatial_bhw.dtype() == torch::kFloat32, "spatial_bhw must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(channel_bc.is_contiguous(), "channel_bc must be contiguous");
    TORCH_CHECK(spatial_bhw.is_contiguous(), "spatial_bhw must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (B,C,H,W)");
    TORCH_CHECK(channel_bc.dim() == 2, "channel_bc must be 2D (B,C)");
    TORCH_CHECK(spatial_bhw.dim() == 2, "spatial_bhw must be 2D (B,HW)");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int HW = H * W;

    TORCH_CHECK(channel_bc.size(0) == B && channel_bc.size(1) == C, "channel_bc shape must be (B,C)");
    TORCH_CHECK(spatial_bhw.size(0) == B && spatial_bhw.size(1) == HW, "spatial_bhw shape must be (B, H*W)");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int threads = 128;
    const int64_t max_blocks = 8192;

    const int64_t n = (int64_t)B * (int64_t)C * (int64_t)HW;

    // Only use vec4 path for HW==49 and full alignment & n%4==0
    const uintptr_t x_ptr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t o_ptr = (uintptr_t)out.data_ptr<float>();
    const bool aligned16 = ((x_ptr | o_ptr) & 0xF) == 0;
    const bool n_div4 = (n & 3LL) == 0;

    if (HW == 49 && aligned16 && n_div4) {
        const int64_t n4 = n >> 2;
        int64_t blocks = ceil_div_i64(n4, (int64_t)threads);
        if (blocks < 1) blocks = 1;
        if (blocks > max_blocks) blocks = max_blocks;

        bam_fused_compact_hw49_vec4_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            channel_bc.data_ptr<float>(),
            spatial_bhw.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C
        );
    } else {
        int64_t blocks = ceil_div_i64(n, (int64_t)threads);
        if (blocks < 1) blocks = 1;
        if (blocks > max_blocks) blocks = max_blocks;

        bam_fused_compact_generic_scalar_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            channel_bc.data_ptr<float>(),
            spatial_bhw.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C, HW
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

bam_fused_cpp_src = r"""
torch::Tensor bam_fused_cuda(torch::Tensor x, torch::Tensor channel, torch::Tensor spatial);
torch::Tensor bam_fused_compact_cuda(torch::Tensor x, torch::Tensor channel_bc, torch::Tensor spatial_bhw);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_bam_opt7",
    cpp_sources=bam_fused_cpp_src,
    cuda_sources=bam_fused_cuda_src,
    functions=["bam_fused_cuda", "bam_fused_compact_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


# ---------------------------
# New model using custom op
# ---------------------------

class ModelNew(nn.Module):
    """BAM with optimized fused elementwise tail.
    Adds a compact-fusion fast path to avoid materializing expanded channel/spatial tensors.
    """
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = ChannelGate(channel)
        self.spatial_attn = SpatialGate(channel)
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        # Compute attention branches using existing modules.
        ch_full = self.channel_attn(x)   # (B,C,H,W) expanded
        sp_full = self.spatial_attn(x)   # (B,C,H,W) expanded (from 1 channel expanded)

        if x.is_cuda and x.dtype == torch.float32:
            # Try compact-fusion path by reconstructing compact tensors cheaply from existing outputs:
            # channel_bc: take any spatial position (all equal due to expand)
            # spatial_bhw: take channel 0 (all channels equal due to expand)
            # This avoids reading ch_full/sp_full during the tail, but note we already computed them.
            # In a fuller rewrite, ChannelGate/SpatialGate would directly return compact tensors.
            B, C, H, W = x.shape
            # Use views that stay contiguous
            ch_bc = ch_full[:, :, 0, 0].contiguous()  # (B,C)
            sp_bhw = sp_full[:, 0, :, :].contiguous().view(B, H * W)  # (B,HW)

            # If contiguity assumptions are met, use compact kernel; else fallback.
            if ch_bc.is_contiguous() and sp_bhw.is_contiguous() and x.is_contiguous():
                return self.custom_ops.bam_fused_compact_cuda(x, ch_bc, sp_bhw)

            return self.custom_ops.bam_fused_cuda(
                x.contiguous(), ch_full.contiguous(), sp_full.contiguous()
            )

        attn = torch.sigmoid(ch_full + sp_full)
        return x + x * attn


# ---------------------------
# Input helpers (kept compatible)
# ---------------------------

batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512]