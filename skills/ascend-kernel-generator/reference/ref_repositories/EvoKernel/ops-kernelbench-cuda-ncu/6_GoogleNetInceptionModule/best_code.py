import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA op: concat along channel dim for 4 NCHW float32 tensors (same N,H,W)
# Optimized v5:
# - Single fused launch (4->1) but routing is PER-PLANE (per output (n,c)), not per element
# - 2D grid: blockIdx.y = plane id, blockIdx.x = HW tile id
# - Vectorization ladder per plane: float4 -> float2 -> scalar with tail-safe handling
# - Avoid div/mod in hot loop; only per-plane math and contiguous indexing
# - __launch_bounds__ to limit registers; explicit kernel launch checks
# -----------------------------------------------------------------------------
concat_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

template<int VEC>
__device__ __forceinline__ bool is_aligned(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & (VEC * sizeof(float) - 1)) == 0);
}

// Plane-wise multi-input concat kernel.
// Each block copies a tile of one output plane (n, c_out) over HW elements.
__global__ __launch_bounds__(256, 2) void concat4_planes_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const float* __restrict__ d,
    float* __restrict__ out,
    int N, int Ca, int Cb, int Cc, int Cd,
    int H, int W
) {
    const int Ctot = Ca + Cb + Cc + Cd;
    const int HW = H * W;

    const int plane = (int)blockIdx.y; // 0 .. N*Ctot-1
    if (plane >= N * Ctot) return;

    const int n = plane / Ctot;
    const int co = plane - n * Ctot;

    // Determine source tensor + channel within that tensor (once per plane)
    const float* __restrict__ src_base;
    int ci;
    if (co < Ca) {
        src_base = a;
        ci = co;
    } else if (co < Ca + Cb) {
        src_base = b;
        ci = co - Ca;
    } else if (co < Ca + Cb + Cc) {
        src_base = c;
        ci = co - (Ca + Cb);
    } else {
        src_base = d;
        ci = co - (Ca + Cb + Cc);
    }

    // Base pointers for this plane
    const int64_t plane_offset_out = ((int64_t)n * (int64_t)Ctot + (int64_t)co) * (int64_t)HW;
    float* __restrict__ outp = out + plane_offset_out;

    // src plane stride depends on which tensor we selected; but tensor layout is NCHW contiguous,
    // so plane offset is (n * Csrc + ci) * HW. We must use correct Csrc:
    int Csrc = (co < Ca) ? Ca : ((co < Ca + Cb) ? Cb : ((co < Ca + Cb + Cc) ? Cc : Cd));
    const int64_t plane_offset_src = ((int64_t)n * (int64_t)Csrc + (int64_t)ci) * (int64_t)HW;
    const float* __restrict__ srcp = src_base + plane_offset_src;

    // Tile along HW
    const int tile = (int)blockIdx.x;
    const int base = tile * blockDim.x + threadIdx.x;

    // Prefer vec4 if both pointers aligned to 16B and HW is large enough; handle tail scalarly.
    // We do not require HW divisible by vector width; we just vectorize the bulk part.
    if (is_aligned<4>(srcp) && is_aligned<4>(outp)) {
        const int HW4 = HW >> 2; // number of float4
        const int base4 = tile * blockDim.x + threadIdx.x;
        // Copy float4 bulk using same grid over float4 units
        for (int i4 = base4; i4 < HW4; i4 += (int)(gridDim.x * blockDim.x)) {
            reinterpret_cast<float4*>(outp)[i4] = reinterpret_cast<const float4*>(srcp)[i4];
        }
        // Tail (0..3 floats)
        const int tail_start = HW4 << 2;
        for (int i = tail_start + base; i < HW; i += (int)(gridDim.x * blockDim.x)) {
            outp[i] = ldg_f32(srcp + i);
        }
        return;
    }

    // Next try vec2 if aligned to 8B
    if (is_aligned<2>(srcp) && is_aligned<2>(outp)) {
        const int HW2 = HW >> 1;
        const int base2 = tile * blockDim.x + threadIdx.x;
        for (int i2 = base2; i2 < HW2; i2 += (int)(gridDim.x * blockDim.x)) {
            reinterpret_cast<float2*>(outp)[i2] = reinterpret_cast<const float2*>(srcp)[i2];
        }
        // Tail (0..1 float)
        const int tail_start = HW2 << 1;
        for (int i = tail_start + base; i < HW; i += (int)(gridDim.x * blockDim.x)) {
            outp[i] = ldg_f32(srcp + i);
        }
        return;
    }

    // Scalar fallback
    for (int i = base; i < HW; i += (int)(gridDim.x * blockDim.x)) {
        outp[i] = ldg_f32(srcp + i);
    }
}

torch::Tensor concat4_nchw_forward_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d
) {
    CHECK_CUDA(a); CHECK_CUDA(b); CHECK_CUDA(c); CHECK_CUDA(d);
    CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b); CHECK_CONTIGUOUS(c); CHECK_CONTIGUOUS(d);
    CHECK_FLOAT(a); CHECK_FLOAT(b); CHECK_FLOAT(c); CHECK_FLOAT(d);

    TORCH_CHECK(a.dim() == 4 && b.dim() == 4 && c.dim() == 4 && d.dim() == 4, "all inputs must be NCHW");
    const int64_t N = a.size(0);
    const int64_t H = a.size(2);
    const int64_t W = a.size(3);

    TORCH_CHECK(b.size(0) == N && c.size(0) == N && d.size(0) == N, "N mismatch");
    TORCH_CHECK(b.size(2) == H && c.size(2) == H && d.size(2) == H, "H mismatch");
    TORCH_CHECK(b.size(3) == W && c.size(3) == W && d.size(3) == W, "W mismatch");

    const int64_t Ca = a.size(1);
    const int64_t Cb = b.size(1);
    const int64_t Cc = c.size(1);
    const int64_t Cd = d.size(1);
    const int64_t Ctot = Ca + Cb + Cc + Cd;

    auto out = torch::empty({N, Ctot, H, W}, a.options());

    const int threads = 256;

    // Grid-x: tiles over HW with some over-subscription to increase MLP.
    // Since each plane is independent, occupancy comes from grid.y = N*Ctot planes.
    const int HW_i = (int)(H * W);
    int tiles = (HW_i + threads - 1) / threads;
    // Slight oversubscription helps latency hiding for small tiles.
    // Clamp to avoid pathological huge grid.x.
    if (tiles < 1) tiles = 1;
    if (tiles > 64) tiles = 64;

    dim3 block(threads, 1, 1);
    dim3 grid((unsigned)tiles, (unsigned)(N * Ctot), 1);

    const auto stream = at::cuda::getDefaultCUDAStream();
    concat4_planes_kernel<<<grid, block, 0, stream>>>(
        (const float*)a.data_ptr<float>(),
        (const float*)b.data_ptr<float>(),
        (const float*)c.data_ptr<float>(),
        (const float*)d.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)N, (int)Ca, (int)Cb, (int)Cc, (int)Cd,
        (int)H, (int)W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

concat_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor concat4_nchw_forward_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_google_net_inception_module_concat4_v5_plane_fused",
    cpp_sources=concat_cpp_source,
    cuda_sources=concat_cuda_source,
    functions=["concat4_nchw_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--ptxas-options=-O3"],
    extra_cflags=["-O3"],
)

# -----------------------------------------------------------------------------
# Model with custom CUDA concat replacing torch.cat in google_net_inception_module
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()

        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.pool(x)
        branch_pool = self.pool_proj(branch_pool)

        if x.is_cuda and x.dtype == torch.float32:
            return custom_ops_lib.concat4_nchw_forward_cuda(
                branch1x1.contiguous(),
                branch3x3.contiguous(),
                branch5x5.contiguous(),
                branch_pool.contiguous(),
            )
        else:
            return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)