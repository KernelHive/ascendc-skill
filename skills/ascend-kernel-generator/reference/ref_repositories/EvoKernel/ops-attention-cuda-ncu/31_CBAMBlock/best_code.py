import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA: further optimized fused tail for CBAM ----
# out = x * ca * sa + x
cbam_fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
#endif

// ---------------- Generic vectorized kernel (fallback) ----------------
__global__ void cbam_fused_generic_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ ca,   // [B*C]
    const float* __restrict__ sa,   // [B*HW]
    float* __restrict__ out,
    int B, int C, int H, int W
) {
    int64_t N = (int64_t)B * C * H * W;
    int64_t N4 = N >> 2;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    int HW = H * W;
    int CHW = C * HW;

    for (int64_t i4 = tid; i4 < N4; i4 += stride) {
        int64_t base = i4 << 2;

        float4 xv = x4[i4];
        float o0, o1, o2, o3;

        // lane 0..3 (keep simple; generic path)
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            int64_t idx = base + lane;
            int b = (int)(idx / CHW);
            int64_t r0 = idx - (int64_t)b * CHW;
            int c = (int)(r0 / HW);
            int hw = (int)(r0 - (int64_t)c * HW);

            float xv_lane = (lane == 0 ? xv.x : (lane == 1 ? xv.y : (lane == 2 ? xv.z : xv.w)));
            float cav = LDG(ca + (int64_t)b * C + c);
            float sav = LDG(sa + (int64_t)b * HW + hw);
            float ov = fmaf(xv_lane, cav * sav, xv_lane);

            if (lane == 0) o0 = ov;
            else if (lane == 1) o1 = ov;
            else if (lane == 2) o2 = ov;
            else o3 = ov;
        }

        out4[i4] = make_float4(o0, o1, o2, o3);
    }
}

__global__ void cbam_fused_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ ca,
    const float* __restrict__ sa,
    float* __restrict__ out,
    int B, int C, int H, int W,
    int64_t start_idx
) {
    int64_t N = (int64_t)B * C * H * W;
    int64_t idx0 = start_idx + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int HW = H * W;
    int CHW = C * HW;

    for (int64_t idx = idx0; idx < N; idx += stride) {
        int b = (int)(idx / CHW);
        int64_t r0 = idx - (int64_t)b * CHW;
        int c = (int)(r0 / HW);
        int hw = (int)(r0 - (int64_t)c * HW);

        float xv = x[idx];
        float cav = LDG(ca + (int64_t)b * C + c);
        float sav = LDG(sa + (int64_t)b * HW + hw);
        out[idx] = fmaf(xv, cav * sav, xv);
    }
}

// ---------------- HW=49, C=512 specialized kernel ----------------
// Mapping for coalescing:
// - Each block handles one batch b and one channel-tile (CT=32 channels).
// - Threads are laid out as (hw, vecLane) where hw in [0,48] and vecLane in [0,7] => 49*8=392 threads.
// - Each thread processes one float4 at a fixed hw for a fixed channel within the 32-channel tile.
// This yields:
// - x/out contiguous along HW for each channel: threads with vecLane fixed and consecutive hw access contiguous memory.
// - SA reused: stage sa[b,hw] into shared once, then reused for all channels in tile.
// - CA reused: each channel's ca loaded once per iteration and reused across HW (within each thread it's fixed).
template<int CT>  // channel tile (must be multiple of 4)
__global__ __launch_bounds__(392, 1)
void cbam_fused_hw49_c512_ct32_kernel(
    const float* __restrict__ x,   // [B,512,49]
    const float* __restrict__ ca,  // [B*512]
    const float* __restrict__ sa,  // [B*49]
    float* __restrict__ out,
    int B
) {
    constexpr int HW = 49;
    constexpr int C = 512;
    constexpr int VEC = 4;
    static_assert(CT % VEC == 0, "CT must be multiple of 4");

    int b = (int)blockIdx.z;
    int tile = (int)blockIdx.y; // 0..(512/CT - 1)
    int c_base = tile * CT;

    // thread mapping
    int tid = threadIdx.x; // 0..391
    int hw = tid & 63;     // use low bits; but only 0..48 valid
    int vlane = tid >> 6;  // 0..6 if 49*8; but tid ranges 0..391 => vlane 0..6, hw 0..63
    // We want 8 vlanes. Remap using division to be safe.
    hw = tid % HW;         // 0..48
    vlane = tid / HW;      // 0..7 (since 392/49=8)
    if (vlane >= (CT / VEC)) return; // CT=32 => CT/VEC=8; ok

    // stage SA for this batch into shared (49 floats)
    __shared__ float sa_s[HW];
    // first 49 threads load
    if (tid < HW) {
        sa_s[tid] = LDG(sa + (int64_t)b * HW + tid);
    }
    __syncthreads();
    float sav = sa_s[hw];

    const float* __restrict__ x_b = x + (int64_t)b * (int64_t)C * HW;
    float* __restrict__ out_b = out + (int64_t)b * (int64_t)C * HW;
    const float* __restrict__ ca_b = ca + (int64_t)b * C;

    // Each thread handles one channel c within tile and one hw, but vectorized across 4 HW elements is not possible
    // because HW=49. Instead vectorize across channels using float4 on x/out at fixed hw would be strided (bad).
    // So we vectorize across HW by using float4 only when hw <= 45 (pack 4 contiguous hw elements).
    // Since each thread has fixed hw, we instead have vlane represent a group of 4 HW starting positions for better coalescing:
    // We keep thread->hw as above (for SA reuse), and manually load/store scalar. The key gain is coalesced across hw.
    // To compensate, we add ILP over channels: each thread processes 2 channels per iteration.
    int c0 = c_base + vlane; // 0..31 but sparse; we'll iterate to cover all CT channels
    // We want each vlane to cover all channels: iterate c = c_base + vlane; c += (CT/VEC)=8
    // But that only covers 4*? No, vlane is 0..7; iter step 8 covers 32 channels exactly.
    // Each thread processes one channel per loop iter; add unroll for 2 iters to increase ILP.
    #pragma unroll
    for (int k = 0; k < (CT / (CT / VEC)); ++k) { // CT=32, CT/(CT/VEC)=4 iterations
        int c = c_base + vlane + k * (CT / VEC);  // vlane + k*8 => covers 0..31
        float cav = LDG(ca_b + c);
        int64_t idx = (int64_t)c * HW + hw;
        float xv = LDG(x_b + idx);
        out_b[idx] = fmaf(xv, cav * sav, xv);
    }
}

torch::Tensor cbam_fused_cuda(torch::Tensor x, torch::Tensor ca, torch::Tensor sa) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(ca.is_cuda(), "ca must be CUDA");
    TORCH_CHECK(sa.is_cuda(), "sa must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(ca.scalar_type() == torch::kFloat32, "ca must be float32");
    TORCH_CHECK(sa.scalar_type() == torch::kFloat32, "sa must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(ca.is_contiguous(), "ca must be contiguous");
    TORCH_CHECK(sa.is_contiguous(), "sa must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");
    TORCH_CHECK(ca.numel() == x.size(0) * x.size(1), "ca must have numel B*C (from [B,C,1,1])");
    TORCH_CHECK(sa.numel() == x.size(0) * x.size(2) * x.size(3), "sa must have numel B*H*W (from [B,1,H,W])");

    const auto B = (int)x.size(0);
    const auto C = (int)x.size(1);
    const auto H = (int)x.size(2);
    const auto W = (int)x.size(3);

    auto out = torch::empty_like(x);

    const int device = x.get_device();
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int grid_cap = sm_count > 0 ? sm_count * 8 : 2048;

    auto stream = at::cuda::getDefaultCUDAStream();

    // Fast path: H*W=49 and C=512
    if ((H * W) == 49 && C == 512) {
        constexpr int CT = 32;
        constexpr int BLOCK = 392; // 49 * (CT/4)=49*8
        dim3 block(BLOCK, 1, 1);
        dim3 grid(1, C / CT, B); // x=1, y=16 tiles, z=B
        cbam_fused_hw49_c512_ct32_kernel<CT><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            ca.data_ptr<float>(),
            sa.data_ptr<float>(),
            out.data_ptr<float>(),
            B
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // Otherwise use vectorized generic kernel when aligned
    int64_t N = (int64_t)B * C * H * W;
    const int block = 256;

    uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    uintptr_t op = (uintptr_t)out.data_ptr<float>();
    bool aligned = ((xp & 0xF) == 0) && ((op & 0xF) == 0);

    if (aligned && (N >= 4)) {
        int64_t N4 = N >> 2;
        int grid = (int)((N4 + block - 1) / block);
        if (grid < 1) grid = 1;
        if (grid > grid_cap) grid = grid_cap;

        cbam_fused_generic_vec4_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            ca.data_ptr<float>(),
            sa.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C, H, W
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        int64_t vec_elems = (N4 << 2);
        if (vec_elems < N) {
            int64_t tail = N - vec_elems;
            int grid_tail = (int)((tail + block - 1) / block);
            if (grid_tail < 1) grid_tail = 1;
            if (grid_tail > grid_cap) grid_tail = grid_cap;

            cbam_fused_scalar_kernel<<<grid_tail, block, 0, stream>>>(
                x.data_ptr<float>(),
                ca.data_ptr<float>(),
                sa.data_ptr<float>(),
                out.data_ptr<float>(),
                B, C, H, W,
                vec_elems
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }

    // Scalar fallback
    {
        int grid2 = (int)((N + block - 1) / block);
        if (grid2 < 1) grid2 = 1;
        if (grid2 > grid_cap) grid2 = grid_cap;

        cbam_fused_scalar_kernel<<<grid2, block, 0, stream>>>(
            x.data_ptr<float>(),
            ca.data_ptr<float>(),
            sa.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C, H, W,
            0
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
}
"""

cbam_fused_cpp_source = r"""
torch::Tensor cbam_fused_cuda(torch::Tensor x, torch::Tensor ca, torch::Tensor sa);
"""

custom_ops_lib = load_inline(
    name="custom_cbam_ops_opt7_hw49_c512_sa_smem_coalesced",
    cpp_sources=cbam_fused_cpp_source,
    cuda_sources=cbam_fused_cuda_source,
    functions=["cbam_fused_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ---- Original submodules kept (conv/pool) ----
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class ModelNew(nn.Module):
    """
    CBAM with optimized fused pointwise tail:
      out = x * CA(x)
      out = out * SA(out)
      out = out + x

    Fused kernel computes:
      out = x * ca * sa + x
    where ca = CA(x), sa = SA(x * ca)
    """
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        if not x.is_cuda:
            residual = x
            out = x * self.ca(x)
            out = out * self.sa(out)
            return out + residual

        x_ = x.contiguous()
        if x_.dtype != torch.float32:
            x_ = x_.float()

        ca = self.ca(x_).contiguous()            # [B,C,1,1]
        x_ca = (x_ * ca).contiguous()            # materialized for SA conv input
        sa = self.sa(x_ca).contiguous()          # [B,1,H,W]

        out = self.custom_ops.cbam_fused_cuda(
            x_,
            ca.view(ca.size(0), ca.size(1)),                 # [B,C]
            sa.view(sa.size(0), sa.size(2) * sa.size(3)),    # [B,HW] flattened
        )
        return out