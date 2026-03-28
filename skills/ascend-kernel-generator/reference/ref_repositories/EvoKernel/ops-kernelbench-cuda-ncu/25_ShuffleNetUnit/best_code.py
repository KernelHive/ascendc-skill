import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------
# Custom CUDA ops:
# 1) channel_shuffle_forward_cuda(x, groups) -> y
# 2) channel_shuffle_add_forward_cuda(x, residual, groups) -> y  (fused shuffle + add)
# Layout: NCHW, float32
# -------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ bool is_aligned_ptr(const void* p, size_t a) {
    return ((uintptr_t)p % a) == 0;
}

template<int GROUPS_SPECIAL>
__device__ __forceinline__ int map_c_in(int c_out, int groups, int cpg) {
    if constexpr (GROUPS_SPECIAL == 3) {
        int c_in_group = c_out / 3;
        int g = c_out - c_in_group * 3;
        return g * cpg + c_in_group;
    } else if constexpr (GROUPS_SPECIAL == 2) {
        int c_in_group = c_out >> 1;
        int g = c_out & 1;
        return g * cpg + c_in_group;
    } else {
        int g = c_out % groups;
        int c_in_group = c_out / groups;
        return g * cpg + c_in_group;
    }
}

template<int GROUPS_SPECIAL, bool FUSE_ADD>
__global__ __launch_bounds__(128, 4)
void channel_shuffle_nchw_tiled_kernel_v6(
    const float* __restrict__ x,
    const float* __restrict__ r,   // residual (optional)
    float* __restrict__ y,
    int N, int C, int H, int W,
    int groups, int cpg
){
    // Map blocks to (n, c_out, h) using a 3D grid:
    // grid.z over N, grid.y over C, grid.x over H tiles
    int n = (int)blockIdx.z;
    int c_out = (int)blockIdx.y;
    int h = (int)blockIdx.x;

    if (n >= N || c_out >= C || h >= H) return;

    int c_in = map_c_in<GROUPS_SPECIAL>(c_out, groups, cpg);

    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t in_base  = ((int64_t)n * (int64_t)C + (int64_t)c_in)  * HW + (int64_t)h * (int64_t)W;
    int64_t out_base = ((int64_t)n * (int64_t)C + (int64_t)c_out) * HW + (int64_t)h * (int64_t)W;

    const float* __restrict__ in_ptr = x + in_base;
    float* __restrict__ out_ptr = y + out_base;
    const float* __restrict__ r_ptr = FUSE_ADD ? (r + out_base) : nullptr;

    int tid = (int)threadIdx.x;

    // Prefer float4 along W when possible
    if (((W & 3) == 0) && is_aligned_ptr(in_ptr, 16) && is_aligned_ptr(out_ptr, 16) &&
        (!FUSE_ADD || is_aligned_ptr(r_ptr, 16))) {

        const float4* __restrict__ in4 = (const float4*)in_ptr;
        float4* __restrict__ out4 = (float4*)out_ptr;
        const float4* __restrict__ r4 = FUSE_ADD ? (const float4*)r_ptr : nullptr;

        int W4 = W >> 2;
        for (int i = tid; i < W4; i += blockDim.x) {
            float4 v = in4[i];
            if constexpr (FUSE_ADD) {
                float4 rv = r4[i];
                v.x += rv.x; v.y += rv.y; v.z += rv.z; v.w += rv.w;
            }
            out4[i] = v;
        }
    } else if (((W & 1) == 0) && is_aligned_ptr(in_ptr, 8) && is_aligned_ptr(out_ptr, 8) &&
               (!FUSE_ADD || is_aligned_ptr(r_ptr, 8))) {

        const float2* __restrict__ in2 = (const float2*)in_ptr;
        float2* __restrict__ out2 = (float2*)out_ptr;
        const float2* __restrict__ r2 = FUSE_ADD ? (const float2*)r_ptr : nullptr;

        int W2 = W >> 1;
        for (int i = tid; i < W2; i += blockDim.x) {
            float2 v = in2[i];
            if constexpr (FUSE_ADD) {
                float2 rv = r2[i];
                v.x += rv.x; v.y += rv.y;
            }
            out2[i] = v;
        }
    } else {
        for (int w = tid; w < W; w += blockDim.x) {
#if __CUDA_ARCH__ >= 350
            float v = __ldg(in_ptr + w);
            if constexpr (FUSE_ADD) v += __ldg(r_ptr + w);
            out_ptr[w] = v;
#else
            float v = in_ptr[w];
            if constexpr (FUSE_ADD) v += r_ptr[w];
            out_ptr[w] = v;
#endif
        }
    }
}

static inline dim3 make_grid(int N, int C, int H) {
    // Use grid.x=H, grid.y=C, grid.z=N; assumes H,C,N fit grid dims (typical conv featuremaps do).
    // For very large C, you could tile C into grid.y and fold, but for ShuffleNet shapes this is safe.
    return dim3((unsigned)H, (unsigned)C, (unsigned)N);
}

torch::Tensor channel_shuffle_forward_cuda(torch::Tensor x, int64_t groups) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(groups > 0, "groups must be > 0");

    const auto N64 = x.size(0);
    const auto C64 = x.size(1);
    const auto H64 = x.size(2);
    const auto W64 = x.size(3);
    TORCH_CHECK(C64 % groups == 0, "channels must be divisible by groups");

    int N = (int)N64, C = (int)C64, H = (int)H64, W = (int)W64;
    int g = (int)groups;
    int cpg = C / g;

    auto y = torch::empty_like(x);

    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 block(128, 1, 1);
    dim3 grid = make_grid(N, C, H);

    const float* xp = (const float*)x.data_ptr<float>();
    float* yp = (float*)y.data_ptr<float>();

    if (g == 3) {
        channel_shuffle_nchw_tiled_kernel_v6<3, false><<<grid, block, 0, stream>>>(xp, nullptr, yp, N, C, H, W, g, cpg);
    } else if (g == 2) {
        channel_shuffle_nchw_tiled_kernel_v6<2, false><<<grid, block, 0, stream>>>(xp, nullptr, yp, N, C, H, W, g, cpg);
    } else {
        channel_shuffle_nchw_tiled_kernel_v6<-1, false><<<grid, block, 0, stream>>>(xp, nullptr, yp, N, C, H, W, g, cpg);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor channel_shuffle_add_forward_cuda(torch::Tensor x, torch::Tensor residual, int64_t groups) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda(), "x and residual must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && residual.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous(), "x and residual must be contiguous (NCHW)");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(groups > 0, "groups must be > 0");

    const auto N64 = x.size(0);
    const auto C64 = x.size(1);
    const auto H64 = x.size(2);
    const auto W64 = x.size(3);
    TORCH_CHECK(C64 % groups == 0, "channels must be divisible by groups");

    int N = (int)N64, C = (int)C64, H = (int)H64, W = (int)W64;
    int g = (int)groups;
    int cpg = C / g;

    auto y = torch::empty_like(x);

    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 block(128, 1, 1);
    dim3 grid = make_grid(N, C, H);

    const float* xp = (const float*)x.data_ptr<float>();
    const float* rp = (const float*)residual.data_ptr<float>();
    float* yp = (float*)y.data_ptr<float>();

    if (g == 3) {
        channel_shuffle_nchw_tiled_kernel_v6<3, true><<<grid, block, 0, stream>>>(xp, rp, yp, N, C, H, W, g, cpg);
    } else if (g == 2) {
        channel_shuffle_nchw_tiled_kernel_v6<2, true><<<grid, block, 0, stream>>>(xp, rp, yp, N, C, H, W, g, cpg);
    } else {
        channel_shuffle_nchw_tiled_kernel_v6<-1, true><<<grid, block, 0, stream>>>(xp, rp, yp, N, C, H, W, g, cpg);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor channel_shuffle_forward_cuda(torch::Tensor x, int64_t groups);
torch::Tensor channel_shuffle_add_forward_cuda(torch::Tensor x, torch::Tensor residual, int64_t groups);
"""

custom_ops_lib = load_inline(
    name="custom_shufflenet_unit_ops_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "channel_shuffle_forward_cuda",
        "channel_shuffle_add_forward_cuda",
    ],
    verbose=False,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        "-lineinfo",
    ],
    extra_cflags=["-O3"],
)


class ChannelShuffleNew(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = int(groups)
        self.custom_ops_lib = custom_ops_lib  # keep reference alive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.groups
        if not x.is_cuda:
            n, c, h, w = x.shape
            cpg = c // g
            return x.view(n, g, cpg, h, w).transpose(1, 2).contiguous().view(n, c, h, w)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        return self.custom_ops_lib.channel_shuffle_forward_cuda(x, g)

    def forward_add(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        g = self.groups
        # CUDA fastpath only if both are float32 contiguous and same shape
        if x.is_cuda and residual.is_cuda:
            if x.dtype != torch.float32:
                x = x.float()
            if residual.dtype != torch.float32:
                residual = residual.float()
            if not x.is_contiguous():
                x = x.contiguous()
            if not residual.is_contiguous():
                residual = residual.contiguous()
            if x.shape == residual.shape:
                return self.custom_ops_lib.channel_shuffle_add_forward_cuda(x, residual, g)

        # fallback
        n, c, h, w = x.shape
        cpg = c // g
        y = x.view(n, g, cpg, h, w).transpose(1, 2).contiguous().view(n, c, h, w)
        return y + residual


class ModelNew(nn.Module):
    """
    ShuffleNet unit with optimized custom CUDA channel shuffle.
    Includes an optional fused shuffle+residual-add fastpath when shortcut is identity.
    """
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib  # keep reference alive

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
            groups=mid_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffleNew(groups)

        self.identity_shortcut = (in_channels == out_channels)
        if self.identity_shortcut:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # If shortcut is identity, fuse shuffle + add to reduce memory traffic by one full tensor op.
        if self.identity_shortcut:
            out = self.shuffle.forward_add(out, x)
        else:
            out = self.shuffle(out)

        out = F.relu(self.bn3(self.conv3(out)))

        if not self.identity_shortcut:
            out = out + self.shortcut(x)

        return out