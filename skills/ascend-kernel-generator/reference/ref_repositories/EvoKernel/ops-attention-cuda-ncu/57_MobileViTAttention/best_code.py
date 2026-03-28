import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Further-optimized MobileViT patch re-layout (ph=pw=7, H=W=49 fast path):
# - Inverse kernel improvement vs baseline:
#   * Precomputed HW->(p,t) mapping in __constant__ memory (no div/mod hot-loop)
#   * Warp-level contiguous HW stores with float4 vectorization when D%4==0
#   * No shared memory staging needed (direct gather then coalesced store by design)
# - Forward kernel kept (already decent), minor alignment-safe vec4 path
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

// For H=W=49, ph=pw=7:
// hw in [0,2401). Map hw -> p in [0,49), t in [0,49).
// We'll pack into uint16: low 6 bits p (0..48), next 6 bits t (0..48).
__constant__ __align__(8) unsigned short c_hw2pt_49[49 * 49];

__global__ void init_hw2pt_49_kernel() {
    int hw = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (hw >= 49 * 49) return;
    int h = hw / 49;
    int w = hw - h * 49;

    int ih  = h / 7;
    int iph = h - ih * 7;
    int iw  = w / 7;
    int ipw = w - iw * 7;

    int p = iph * 7 + ipw;      // 0..48
    int t = ih * 7 + iw;        // 0..48
    unsigned short packed = (unsigned short)((p & 63) | ((t & 63) << 6));
    // write to global temp? Can't write to constant from device. This kernel is unused.
}

// Host-side initializer for constant memory LUT
static inline void init_hw2pt_49(cudaStream_t stream) {
    static bool inited = false;
    if (inited) return;

    unsigned short host_lut[49 * 49];
    for (int hw = 0; hw < 49 * 49; ++hw) {
        int h = hw / 49;
        int w = hw - h * 49;

        int ih  = h / 7;
        int iph = h - ih * 7;
        int iw  = w / 7;
        int ipw = w - iw * 7;

        int p = iph * 7 + ipw;
        int t = ih * 7 + iw;

        host_lut[hw] = (unsigned short)((p & 63) | ((t & 63) << 6));
    }
    cudaMemcpyToSymbolAsync(c_hw2pt_49, host_lut, sizeof(host_lut), 0, cudaMemcpyHostToDevice, stream);
    inited = true;
}

__device__ __forceinline__ void decode_p7(int p, int &iph, int &ipw) {
    iph = p / 7;
    ipw = p - iph * 7;
}

__device__ __forceinline__ void decode_t(int t, int nw, int &ih, int &iw) {
    ih = t / nw;
    iw = t - ih * nw;
}

// Forward: NCHW (B,D,H,W) -> patches (B,P,T,D), D contiguous.
// Grid: (tile_t, p, b). Block: 256 threads mapped along D.
template<bool VEC4>
__global__ __launch_bounds__(256, 4)
void nchw_to_bptd7_kernel(
    const float* __restrict__ x, // (B,D,H,W)
    float* __restrict__ y,       // (B,P,T,D)
    int B, int D, int H, int W,
    int nh, int nw, int T
) {
    int tile_t = (int)blockIdx.x;
    int p      = (int)blockIdx.y;
    int b      = (int)blockIdx.z;

    constexpr int TILE_T = 4;
    int t0 = tile_t * TILE_T;
    if (t0 >= T) return;

    int iph, ipw;
    decode_p7(p, iph, ipw);

    int tid = (int)threadIdx.x;

    int64_t stride = (int64_t)H * (int64_t)W;

    if constexpr (VEC4) {
        int D4 = D >> 2;
        #pragma unroll
        for (int tt = 0; tt < TILE_T; ++tt) {
            int t = t0 + tt;
            if (t >= T) break;
            int ih, iw;
            decode_t(t, nw, ih, iw);
            int h = ih * 7 + iph;
            int w = iw * 7 + ipw;

            int64_t x_hw = (int64_t)h * (int64_t)W + (int64_t)w;
            int64_t x_base = ((int64_t)b * (int64_t)D) * stride + x_hw;
            int64_t y_base = (((int64_t)b * (int64_t)49 + (int64_t)p) * (int64_t)T + (int64_t)t) * (int64_t)D;

            float4* y4 = reinterpret_cast<float4*>(y + y_base);

            for (int d4 = tid; d4 < D4; d4 += 256) {
                int d = d4 << 2;
                const float* x0 = x + x_base + (int64_t)d * stride;
                float4 v;
                v.x = LDG(x0 + 0 * stride);
                v.y = LDG(x0 + 1 * stride);
                v.z = LDG(x0 + 2 * stride);
                v.w = LDG(x0 + 3 * stride);
                y4[d4] = v;
            }
        }
    } else {
        #pragma unroll
        for (int tt = 0; tt < TILE_T; ++tt) {
            int t = t0 + tt;
            if (t >= T) break;
            int ih, iw;
            decode_t(t, nw, ih, iw);
            int h = ih * 7 + iph;
            int w = iw * 7 + ipw;

            int64_t x_hw = (int64_t)h * (int64_t)W + (int64_t)w;
            int64_t x_base = ((int64_t)b * (int64_t)D) * stride + x_hw;
            int64_t y_base = (((int64_t)b * (int64_t)49 + (int64_t)p) * (int64_t)T + (int64_t)t) * (int64_t)D;

            for (int d = tid; d < D; d += 256) {
                y[y_base + d] = LDG(x + x_base + (int64_t)d * stride);
            }
        }
    }
}

// Fully specialized inverse for H=W=49, ph=pw=7, T=49, P=49.
// Strategy:
// - Make stores contiguous and vectorized along HW for each (b, d4).
// - Each block covers: one batch b, one d4 tile (float4 channels), and a HW segment.
// - Threads: 128 (4 warps). Each warp writes a contiguous HW strip with float4 stores.
template<int HW_TILE>
__global__ __launch_bounds__(128, 6)
void bptd7_to_nchw49_vec4_kernel(
    const float* __restrict__ y, // (B,49,49,D)
    float* __restrict__ x,       // (B,D,49,49)
    int D
) {
    constexpr int H = 49;
    constexpr int W = 49;
    constexpr int HW = H * W;
    constexpr int T = 49;
    constexpr int P = 49;

    int hw_tile = (int)blockIdx.x;   // ceil(HW/HW_TILE)
    int d4_tile = (int)blockIdx.y;   // ceil((D/4)/D4_TILE) but we use 1 d4 per warp-group iteration
    int b       = (int)blockIdx.z;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int hw0 = hw_tile * HW_TILE;

    // We map blockIdx.y as a "base d4" chunk of 4 warps * 1 d4 each per inner loop step.
    // D4 = D/4. Each warp handles one d4 index at a time.
    int D4 = D >> 2;
    int base_d4 = d4_tile * 4;

    // Each lane writes one hw element (contiguous in HW), repeated for HW_TILE/32 segments.
    #pragma unroll
    for (int seg = 0; seg < (HW_TILE / 32); ++seg) {
        int hw = hw0 + seg * 32 + lane;
        if (hw >= HW) continue;

        // decode p,t from constant LUT
        unsigned short packed = c_hw2pt_49[hw];
        int p = (int)(packed & 63);
        int t = (int)((packed >> 6) & 63);

        int d4 = base_d4 + warp;
        if (d4 < D4) {
            int d = d4 << 2;

            int64_t y_base = (((int64_t)b * (int64_t)P + (int64_t)p) * (int64_t)T + (int64_t)t) * (int64_t)D + (int64_t)d;
            float4 v = *reinterpret_cast<const float4*>(y + y_base);

            // x is contiguous in HW for fixed (b,d..d+3)
            int64_t x_base = ((int64_t)b * (int64_t)D + (int64_t)d) * (int64_t)HW + (int64_t)hw;

            // store each component with stride HW (since x is [B,D,HW])
            x[x_base + 0 * (int64_t)HW] = v.x;
            x[x_base + 1 * (int64_t)HW] = v.y;
            x[x_base + 2 * (int64_t)HW] = v.z;
            x[x_base + 3 * (int64_t)HW] = v.w;
        }
    }
}

// Scalar fallback for D%4!=0 but still H=W=49 case: keep stores contiguous by HW for each d.
template<int HW_TILE>
__global__ __launch_bounds__(128, 6)
void bptd7_to_nchw49_scalar_kernel(
    const float* __restrict__ y, // (B,49,49,D)
    float* __restrict__ x,       // (B,D,49,49)
    int D
) {
    constexpr int H = 49;
    constexpr int W = 49;
    constexpr int HW = H * W;
    constexpr int T = 49;
    constexpr int P = 49;

    int hw_tile = (int)blockIdx.x;
    int d_tile  = (int)blockIdx.y; // one d per warp at a time
    int b       = (int)blockIdx.z;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int hw0 = hw_tile * HW_TILE;
    int base_d = d_tile * 4; // 4 warps -> 4 d's

    #pragma unroll
    for (int seg = 0; seg < (HW_TILE / 32); ++seg) {
        int hw = hw0 + seg * 32 + lane;
        if (hw >= HW) continue;

        unsigned short packed = c_hw2pt_49[hw];
        int p = (int)(packed & 63);
        int t = (int)((packed >> 6) & 63);

        int d = base_d + warp;
        if (d < D) {
            int64_t y_base = (((int64_t)b * (int64_t)P + (int64_t)p) * (int64_t)T + (int64_t)t) * (int64_t)D + (int64_t)d;
            float v = LDG(y + y_base);

            int64_t x_idx = ((int64_t)b * (int64_t)D + (int64_t)d) * (int64_t)HW + (int64_t)hw;
            x[x_idx] = v;
        }
    }
}

// Generic fallback baseline inverse: grid (w,h,b), strided stores across D.
template<bool VEC4>
__global__ __launch_bounds__(256, 4)
void bptd7_to_nchw_kernel_baseline(
    const float* __restrict__ y, // (B,P,T,D)
    float* __restrict__ x,       // (B,D,H,W)
    int B, int D, int H, int W,
    int nh, int nw, int T
) {
    int w = (int)blockIdx.x;
    int h = (int)blockIdx.y;
    int b = (int)blockIdx.z;
    if (w >= W || h >= H) return;

    int ih = h / 7;
    int iph = h - ih * 7;
    int iw = w / 7;
    int ipw = w - iw * 7;

    int p = iph * 7 + ipw;
    int t = ih * nw + iw;

    int tid = (int)threadIdx.x;
    int64_t stride = (int64_t)H * (int64_t)W;
    int64_t x_hw = (int64_t)h * (int64_t)W + (int64_t)w;
    int64_t x_base = ((int64_t)b * (int64_t)D) * stride + x_hw;

    int64_t y_base = (((int64_t)b * (int64_t)49 + (int64_t)p) * (int64_t)T + (int64_t)t) * (int64_t)D;

    if constexpr (VEC4) {
        int D4 = D >> 2;
        const float4* y4 = (const float4*)(y + y_base);
        for (int d4 = tid; d4 < D4; d4 += 256) {
            float4 v = y4[d4];
            int d = d4 << 2;
            float* x0 = x + x_base + (int64_t)d * stride;
            x0[0 * stride] = v.x;
            x0[1 * stride] = v.y;
            x0[2 * stride] = v.z;
            x0[3 * stride] = v.w;
        }
    } else {
        for (int d = tid; d < D; d += 256) {
            x[x_base + (int64_t)d * stride] = y[y_base + d];
        }
    }
}

torch::Tensor mobilevit_nchw_to_bptd_cuda(torch::Tensor x, int64_t ph, int64_t pw) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be (B,D,H,W)");
    TORCH_CHECK(ph == 7 && pw == 7, "Only ph=pw=7 supported in this optimized build");

    int B = (int)x.size(0);
    int D = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    TORCH_CHECK(H % 7 == 0 && W % 7 == 0, "H/W must be divisible by 7");

    int nh = H / 7;
    int nw = W / 7;
    int T = nh * nw;
    int P = 49;

    auto y = torch::empty({B, P, T, D}, x.options());

    bool vec4 = (D % 4 == 0) &&
                (((uintptr_t)x.data_ptr<float>() & 0xF) == 0) &&
                (((uintptr_t)y.data_ptr<float>() & 0xF) == 0);

    dim3 block(256);
    int tile_t = (T + 4 - 1) / 4;
    dim3 grid((unsigned)tile_t, (unsigned)P, (unsigned)B);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (vec4) {
        nchw_to_bptd7_kernel<true><<<grid, block, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, D, H, W, nh, nw, T
        );
    } else {
        nchw_to_bptd7_kernel<false><<<grid, block, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, D, H, W, nh, nw, T
        );
    }
    return y;
}

torch::Tensor mobilevit_bptd_to_nchw_cuda(torch::Tensor y, int64_t H, int64_t W, int64_t ph, int64_t pw) {
    TORCH_CHECK(y.is_cuda(), "y must be CUDA");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(y.dim() == 4, "y must be (B,P,T,D)");
    TORCH_CHECK(ph == 7 && pw == 7, "Only ph=pw=7 supported in this optimized build");
    TORCH_CHECK(H % 7 == 0 && W % 7 == 0, "H/W must be divisible by 7");
    TORCH_CHECK((int)y.size(1) == 49, "P must be 49 for ph=pw=7");

    int B = (int)y.size(0);
    int T = (int)y.size(2);
    int D = (int)y.size(3);

    int nh = (int)(H / 7);
    int nw = (int)(W / 7);
    TORCH_CHECK(nh * nw == T, "T must equal (H/7)*(W/7)");

    auto x = torch::empty({B, D, (int)H, (int)W}, y.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Initialize LUT once (on first call) on this stream
    if (H == 49 && W == 49 && T == 49) {
        init_hw2pt_49(stream);
    }

    // Fast path specialization for H=W=49, T=49.
    if (H == 49 && W == 49 && T == 49) {
        constexpr int HW_TILE = 128; // 4 warps * 32 lanes = 128 contiguous hw per block
        dim3 block(128);

        int HW = 49 * 49;
        int grid_x = (HW + HW_TILE - 1) / HW_TILE;

        bool vec4 = (D % 4 == 0) &&
                    (((uintptr_t)y.data_ptr<float>() & 0xF) == 0) &&
                    (((uintptr_t)x.data_ptr<float>() & 0xF) == 0);

        if (vec4) {
            int D4 = D >> 2;
            int grid_y = (D4 + 4 - 1) / 4; // 4 warps per block => 4 d4 per block
            dim3 grid((unsigned)grid_x, (unsigned)grid_y, (unsigned)B);
            bptd7_to_nchw49_vec4_kernel<HW_TILE><<<grid, block, 0, stream>>>(
                (const float*)y.data_ptr<float>(),
                (float*)x.data_ptr<float>(),
                D
            );
        } else {
            int grid_y = (D + 4 - 1) / 4; // 4 warps => 4 d per block
            dim3 grid((unsigned)grid_x, (unsigned)grid_y, (unsigned)B);
            bptd7_to_nchw49_scalar_kernel<HW_TILE><<<grid, block, 0, stream>>>(
                (const float*)y.data_ptr<float>(),
                (float*)x.data_ptr<float>(),
                D
            );
        }
        return x;
    }

    // Generic fallback baseline kernel.
    bool vec4 = (D % 4 == 0);
    dim3 block(256);
    dim3 grid((unsigned)W, (unsigned)H, (unsigned)B);

    if (vec4) {
        bptd7_to_nchw_kernel_baseline<true><<<grid, block, 0, stream>>>(
            (const float*)y.data_ptr<float>(),
            (float*)x.data_ptr<float>(),
            B, D, (int)H, (int)W, nh, nw, T
        );
    } else {
        bptd7_to_nchw_kernel_baseline<false><<<grid, block, 0, stream>>>(
            (const float*)y.data_ptr<float>(),
            (float*)x.data_ptr<float>(),
            B, D, (int)H, (int)W, nh, nw, T
        );
    }
    return x;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor mobilevit_nchw_to_bptd_cuda(torch::Tensor x, int64_t ph, int64_t pw);
torch::Tensor mobilevit_bptd_to_nchw_cuda(torch::Tensor y, int64_t H, int64_t W, int64_t ph, int64_t pw);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mobile_vi_t_attention_opt7",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "mobilevit_nchw_to_bptd_cuda",
        "mobilevit_bptd_to_nchw_cuda",
    ],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        b, p, n, hd = qkv[0].shape[0], qkv[0].shape[1], qkv[0].shape[2], self.heads
        q, k, v = [t.reshape(b, p, n, hd, -1).permute(0, 1, 3, 2, 4) for t in qkv]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, p, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class ModelNew(nn.Module):
    """
    MobileViT module with optimized CUDA patch re-layout kernels:
    NCHW -> (B,P,T,D) and inverse with a LUT-based, coalesced-store fast path
    for H=W=49, patch_size=7.
    """
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        x_in = x.clone()

        y = self.conv2(self.conv1(x))  # (B, D, H, W)

        if y.dtype != torch.float32:
            y = y.float()
        y = y.contiguous()

        B, D, H, W = y.shape
        ph, pw = self.ph, self.pw

        y = self.custom_ops_lib.mobilevit_nchw_to_bptd_cuda(y, ph, pw)
        y = self.trans(y)

        if y.dtype != torch.float32:
            y = y.float()
        y = y.contiguous()

        y = self.custom_ops_lib.mobilevit_bptd_to_nchw_cuda(y, H, W, ph, pw)

        y = self.conv3(y)
        y = torch.cat([x_in, y], 1)
        y = self.conv4(y)
        return y