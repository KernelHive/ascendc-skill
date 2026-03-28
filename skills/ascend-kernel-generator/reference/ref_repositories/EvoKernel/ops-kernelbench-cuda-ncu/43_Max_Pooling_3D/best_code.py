import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

maxpool3d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline int64_t div_floor_i64(int64_t a, int64_t b) { return a / b; }

__device__ __forceinline__ float fmaxf2(float a, float b) { return a > b ? a : b; }

__device__ __forceinline__ float ro_load_f32(const float* p) {
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}

// ----------------------------
// Generic kernel (fallback)
// ----------------------------
__global__ void maxpool3d_forward_kernel_generic_ncdhw_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * outD * outH * outW;
    if (tid >= total) return;

    int64_t t = tid;
    int w_out = (int)(t % outW); t /= outW;
    int h_out = (int)(t % outH); t /= outH;
    int d_out = (int)(t % outD); t /= outD;
    int c = (int)(t % C); t /= C;
    int n = (int)t;

    int d_start = d_out * sD - pD;
    int h_start = h_out * sH - pH;
    int w_start = w_out * sW - pW;

    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t DHW = (int64_t)D * HW;
    const int64_t base_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * DHW;

    float maxv = -INFINITY;

#pragma unroll 1
    for (int kd = 0; kd < kD; ++kd) {
        int d_in = d_start + kd;
        if ((unsigned)d_in >= (unsigned)D) continue;
        int64_t base_d = base_nc + (int64_t)d_in * HW;
#pragma unroll 1
        for (int kh = 0; kh < kH; ++kh) {
            int h_in = h_start + kh;
            if ((unsigned)h_in >= (unsigned)H) continue;
            int64_t base_h = base_d + (int64_t)h_in * (int64_t)W;
#pragma unroll 1
            for (int kw = 0; kw < kW; ++kw) {
                int w_in = w_start + kw;
                if ((unsigned)w_in >= (unsigned)W) continue;
                float v = ro_load_f32(x + base_h + (int64_t)w_in);
                maxv = fmaxf2(maxv, v);
            }
        }
    }

    y[tid] = maxv;
}

// ----------------------------
// Fast kernel: k=3, s=2, p=1 specialized
// Tile outputs: TILE^3 per block for one (n,c)
// Input brick: (2*TILE+1)^3
// Threads: 128. Cooperative scalar load over brick, then each thread computes
// up to 2 outputs in a strided loop (for TILE=6 => outputs=216).
// ----------------------------
template<int TILE>
__global__ __launch_bounds__(128, 3)
void maxpool3d_forward_k3s2p1_tile_f32_v2(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int tileD
) {
    constexpr int s = 2;
    constexpr int p = 1;
    constexpr int BRICK = 2 * TILE + 1;

    extern __shared__ float sbrick[]; // BRICK^3 floats

    // grid mapping:
    // blockIdx.x = tile_w, blockIdx.y = tile_h, blockIdx.z = (nc * tileD + tile_d)
    const int tw = (int)blockIdx.x;
    const int th = (int)blockIdx.y;
    const int z = (int)blockIdx.z;

    const int td = z % tileD;
    const int nc = z / tileD;
    const int n = nc / C;
    const int c = nc - n * C;
    if (n >= N) return;

    const int ow0 = tw * TILE;
    const int oh0 = th * TILE;
    const int od0 = td * TILE;

    const int w0 = ow0 * s - p;
    const int h0 = oh0 * s - p;
    const int d0 = od0 * s - p;

    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t DHW = (int64_t)D * HW;
    const int64_t base_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * DHW;

    // Cooperative brick load: linear scalar loop (avoids mismatched vector tails).
    const int brick_elems = BRICK * BRICK * BRICK;
    for (int t = (int)threadIdx.x; t < brick_elems; t += (int)blockDim.x) {
        int tmp = t;
        int bw = tmp % BRICK; tmp /= BRICK;
        int bh = tmp % BRICK; tmp /= BRICK;
        int bd = tmp;

        int wi = w0 + bw;
        int hi = h0 + bh;
        int di = d0 + bd;

        float v = -INFINITY;
        if ((unsigned)di < (unsigned)D && (unsigned)hi < (unsigned)H && (unsigned)wi < (unsigned)W) {
            int64_t off = base_nc + (int64_t)di * HW + (int64_t)hi * (int64_t)W + (int64_t)wi;
            v = ro_load_f32(x + off);
        }
        sbrick[t] = v;
    }
    __syncthreads();

    // Precompute output strides and base output pointer for this (n,c)
    const int64_t outHW = (int64_t)outH * (int64_t)outW;
    const int64_t outDHW = (int64_t)outD * outHW;
    const int64_t ybase_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * outDHW;

    const int outputs = TILE * TILE * TILE;
    // Each thread computes multiple outputs if outputs > blockDim.x (true for TILE=6, 216>128)
    for (int t = (int)threadIdx.x; t < outputs; t += (int)blockDim.x) {
        int tmp = t;
        int owOff = tmp % TILE; tmp /= TILE;
        int ohOff = tmp % TILE; tmp /= TILE;
        int odOff = tmp;

        const int ow = ow0 + owOff;
        const int oh = oh0 + ohOff;
        const int od = od0 + odOff;

        if ((unsigned)ow < (unsigned)outW && (unsigned)oh < (unsigned)outH && (unsigned)od < (unsigned)outD) {
            const int bw0 = 2 * owOff;
            const int bh0 = 2 * ohOff;
            const int bd0 = 2 * odOff;

            float maxv = -INFINITY;

#pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
#pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    const int bd = bd0 + kd;
                    const int bh = bh0 + kh;
                    const int base = bd * (BRICK * BRICK) + bh * BRICK + bw0;

                    float v0 = sbrick[base + 0];
                    float v1 = sbrick[base + 1];
                    float v2 = sbrick[base + 2];
                    maxv = fmaxf2(maxv, v0);
                    maxv = fmaxf2(maxv, v1);
                    maxv = fmaxf2(maxv, v2);
                }
            }

            const int64_t yoff = ybase_nc + (int64_t)od * outHW + (int64_t)oh * (int64_t)outW + (int64_t)ow;
            y[yoff] = maxv;
        }
    }
}

torch::Tensor maxpool3d_forward_cuda(
    torch::Tensor x,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    bool ceil_mode
) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 5, "Expected input of shape (N,C,D,H,W)");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    TORCH_CHECK(kD > 0 && kH > 0 && kW > 0, "kernel sizes must be > 0");
    TORCH_CHECK(sD > 0 && sH > 0 && sW > 0, "strides must be > 0");
    TORCH_CHECK(pD >= 0 && pH >= 0 && pW >= 0, "padding must be >= 0");

    auto out_size = [&](int64_t in, int64_t k, int64_t s, int64_t p) -> int64_t {
        int64_t num = in + 2 * p - (k - 1) - 1; // dilation=1
        if (ceil_mode) return div_floor_i64(num + s - 1, s) + 1;
        return div_floor_i64(num, s) + 1;
    };

    int64_t outD = out_size(D, kD, sD, pD);
    int64_t outH = out_size(H, kH, sH, pH);
    int64_t outW = out_size(W, kW, sW, pW);
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "Invalid output size computed");

    auto y = torch::empty({N, C, outD, outH, outW}, x.options());

    const bool fast =
        (kD == 3 && kH == 3 && kW == 3) &&
        (sD == 2 && sH == 2 && sW == 2) &&
        (pD == 1 && pH == 1 && pW == 1);

    if (fast) {
        constexpr int TILE = 6; // larger tile for better reuse / fewer global loads per output
        const int tileW = ((int)outW + TILE - 1) / TILE;
        const int tileH = ((int)outH + TILE - 1) / TILE;
        const int tileD = ((int)outD + TILE - 1) / TILE;

        dim3 grid((unsigned)tileW, (unsigned)tileH, (unsigned)((int)N * (int)C * tileD));
        dim3 block(128, 1, 1);

        constexpr int BRICK = 2 * TILE + 1; // 13
        const size_t shmem = (size_t)(BRICK * BRICK * BRICK) * sizeof(float);

        maxpool3d_forward_k3s2p1_tile_f32_v2<TILE><<<grid, block, shmem>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)C, (int)D, (int)H, (int)W,
            (int)outD, (int)outH, (int)outW,
            tileD
        );
    } else {
        int64_t total = N * C * outD * outH * outW;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);

        maxpool3d_forward_kernel_generic_ncdhw_f32<<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)C, (int)D, (int)H, (int)W,
            (int)outD, (int)outH, (int)outW,
            (int)kD, (int)kH, (int)kW,
            (int)sD, (int)sH, (int)sW,
            (int)pD, (int)pH, (int)pW
        );
    }

    return y;
}
"""

maxpool3d_cpp_source = r"""
torch::Tensor maxpool3d_forward_cuda(
    torch::Tensor x,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    bool ceil_mode
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_maxpool3d_v8",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_cuda_source,
    functions=["maxpool3d_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

# ----------------------------
# New model using custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    MaxPool3d replaced with a custom CUDA kernel.
    Assumptions:
      - input is CUDA, float32, contiguous, shape (N,C,D,H,W)
      - dilation is 1
      - return_indices=False
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        if stride is None:
            stride = kernel_size
        if dilation != 1:
            raise ValueError("This custom kernel implementation supports dilation=1 only.")
        if return_indices:
            raise ValueError("This custom kernel implementation does not return indices.")

        self.kD = int(kernel_size)
        self.kH = int(kernel_size)
        self.kW = int(kernel_size)

        self.sD = int(stride)
        self.sH = int(stride)
        self.sW = int(stride)

        self.pD = int(padding)
        self.pH = int(padding)
        self.pW = int(padding)

        self.ceil_mode = bool(ceil_mode)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects a CUDA tensor input.")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        return self.custom_ops_lib.maxpool3d_forward_cuda(
            x,
            self.kD, self.kH, self.kW,
            self.sD, self.sH, self.sW,
            self.pD, self.pH, self.pW,
            self.ceil_mode,
        )