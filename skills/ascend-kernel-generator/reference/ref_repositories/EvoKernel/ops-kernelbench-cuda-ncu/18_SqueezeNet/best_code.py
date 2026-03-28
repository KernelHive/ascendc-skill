import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------
# Custom CUDA for FireModule forward (squeeze+relu, expand1x1+relu, expand3x3+relu, concat)
# -----------------------------
fire_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

__device__ __forceinline__ float relu_f(float v) { return v > 0.f ? v : 0.f; }

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ld_g(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ld_g(const float* p) { return *p; }
#endif

template<bool HAS_BIAS>
__global__ __launch_bounds__(256, 2)
void squeeze1x1_relu_kernel(
    const float* __restrict__ x,     // [N,Cin,H,W]
    const float* __restrict__ w_sq,  // [Csq,Cin]
    const float* __restrict__ b_sq,  // [Csq] (if HAS_BIAS)
    float* __restrict__ s,           // [N,Csq,H,W]
    int N, int Cin, int H, int W, int Csq
){
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int HW = H * W;
    int total = N * Csq * HW;
    if (tid >= total) return;

    int p = tid % HW;
    int tmp = tid / HW;
    int oc = tmp % Csq;
    int n  = tmp / Csq;

    int x_base = (n * Cin) * HW + p;

    float acc = HAS_BIAS ? ld_g(b_sq + oc) : 0.f;
    const float* wp = w_sq + (long)oc * (long)Cin;
    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        float xv = ld_g(x + x_base + (long)ic * (long)HW);
        float wv = ld_g(wp + ic);
        acc = fmaf(xv, wv, acc);
    }

    s[(n * Csq + oc) * HW + p] = relu_f(acc);
}

template<int TILE_W, int TILE_H, int OC_BLK>
__global__ __launch_bounds__(TILE_W*TILE_H, 2)
void expand1x1_ocblk(
    const float* __restrict__ s,     // [N,Csq,H,W]
    const float* __restrict__ w_e1,  // [Ce1,Csq]
    const float* __restrict__ b_e1,  // [Ce1] or nullptr
    float* __restrict__ y,           // [N,CeT,H,W]
    int N, int Csq, int H, int W,
    int Ce1, int CeT
){
    int oc_tiles = (Ce1 + OC_BLK - 1) / OC_BLK;
    int z = (int)blockIdx.z;
    int n = z / oc_tiles;
    int ocb = z - n * oc_tiles;
    if (n >= N) return;

    int oc0 = ocb * OC_BLK;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int ox = (int)blockIdx.x * TILE_W + tx;
    int oy = (int)blockIdx.y * TILE_H + ty;
    if ((unsigned)ox >= (unsigned)W || (unsigned)oy >= (unsigned)H) return;

    int HW = H * W;
    int out_hw = oy * W + ox;

    int s_n_base = n * Csq * HW;
    int y_n_base = n * CeT * HW;

    float acc[OC_BLK];
    #pragma unroll
    for (int i = 0; i < OC_BLK; ++i) {
        int oc = oc0 + i;
        acc[i] = (oc < Ce1 && b_e1) ? ld_g(b_e1 + oc) : 0.f;
    }

    #pragma unroll 1
    for (int csq = 0; csq < Csq; ++csq) {
        float sv = ld_g(s + s_n_base + csq * HW + out_hw);
        #pragma unroll
        for (int i = 0; i < OC_BLK; ++i) {
            int oc = oc0 + i;
            if (oc < Ce1) {
                float wv = ld_g(w_e1 + (long)oc * (long)Csq + csq);
                acc[i] = fmaf(sv, wv, acc[i]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < OC_BLK; ++i) {
        int oc = oc0 + i;
        if (oc < Ce1) {
            y[y_n_base + oc * HW + out_hw] = relu_f(acc[i]);
        }
    }
}

template<int TILE_W, int TILE_H, int OC_BLK>
__global__ __launch_bounds__(TILE_W*TILE_H, 2)
void expand3x3_ocblk_smem(
    const float* __restrict__ s,     // [N,Csq,H,W]
    const float* __restrict__ w_e3,  // [Ce3,Csq,3,3] flattened
    const float* __restrict__ b_e3,  // [Ce3] or nullptr
    float* __restrict__ y,           // [N,CeT,H,W]
    int N, int Csq, int H, int W,
    int Ce1, int Ce3, int CeT
){
    int oc_tiles = (Ce3 + OC_BLK - 1) / OC_BLK;
    int z = (int)blockIdx.z;
    int n = z / oc_tiles;
    int ocb = z - n * oc_tiles;
    if (n >= N) return;

    int oc30 = ocb * OC_BLK;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    int ox = (int)blockIdx.x * TILE_W + tx;
    int oy = (int)blockIdx.y * TILE_H + ty;

    int HW = H * W;
    int s_n_base = n * Csq * HW;
    int y_n_base = n * CeT * HW;

    bool in_bounds = ((unsigned)ox < (unsigned)W) && ((unsigned)oy < (unsigned)H);
    int out_hw = oy * W + ox;

    float acc[OC_BLK];
    #pragma unroll
    for (int i = 0; i < OC_BLK; ++i) {
        int oc3 = oc30 + i;
        acc[i] = (oc3 < Ce3 && b_e3) ? ld_g(b_e3 + oc3) : 0.f;
    }

    extern __shared__ float sh[];
    const int SHW = TILE_W + 2;
    const int SHH = TILE_H + 2;

    auto sh_at = [&](int yy, int xx) -> float& {
        return sh[yy * SHW + xx];
    };

    #pragma unroll 1
    for (int csq = 0; csq < Csq; ++csq) {
        const float* sp = s + s_n_base + csq * HW;

        int lin = ty * TILE_W + tx;
        int tot = SHH * SHW;
        for (int idx = lin; idx < tot; idx += TILE_W * TILE_H) {
            int yy = idx / SHW;
            int xx = idx - yy * SHW;
            int iy = (int)blockIdx.y * TILE_H + (yy - 1);
            int ix = (int)blockIdx.x * TILE_W + (xx - 1);
            float v = 0.f;
            if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
                v = ld_g(sp + iy * W + ix);
            }
            sh_at(yy, xx) = v;
        }
        __syncthreads();

        if (in_bounds) {
            int sy = ty + 1;
            int sx = tx + 1;

            float t0 = sh_at(sy - 1, sx - 1);
            float t1 = sh_at(sy - 1, sx    );
            float t2 = sh_at(sy - 1, sx + 1);
            float c0 = sh_at(sy    , sx - 1);
            float c1 = sh_at(sy    , sx    );
            float c2 = sh_at(sy    , sx + 1);
            float b0 = sh_at(sy + 1, sx - 1);
            float b1 = sh_at(sy + 1, sx    );
            float b2 = sh_at(sy + 1, sx + 1);

            #pragma unroll
            for (int i = 0; i < OC_BLK; ++i) {
                int oc3 = oc30 + i;
                if (oc3 < Ce3) {
                    const float* wp = w_e3 + (long)oc3 * (long)Csq * 9L + (long)csq * 9L;
                    acc[i] = fmaf(t0, ld_g(wp + 0), acc[i]);
                    acc[i] = fmaf(t1, ld_g(wp + 1), acc[i]);
                    acc[i] = fmaf(t2, ld_g(wp + 2), acc[i]);
                    acc[i] = fmaf(c0, ld_g(wp + 3), acc[i]);
                    acc[i] = fmaf(c1, ld_g(wp + 4), acc[i]);
                    acc[i] = fmaf(c2, ld_g(wp + 5), acc[i]);
                    acc[i] = fmaf(b0, ld_g(wp + 6), acc[i]);
                    acc[i] = fmaf(b1, ld_g(wp + 7), acc[i]);
                    acc[i] = fmaf(b2, ld_g(wp + 8), acc[i]);
                }
            }
        }
        __syncthreads();
    }

    if (in_bounds) {
        #pragma unroll
        for (int i = 0; i < OC_BLK; ++i) {
            int oc3 = oc30 + i;
            if (oc3 < Ce3) {
                int oc = Ce1 + oc3;
                y[y_n_base + oc * HW + out_hw] = relu_f(acc[i]);
            }
        }
    }
}

static inline void cuda_check_last_error() {
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
}

torch::Tensor fire_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_sq, torch::Tensor b_sq,
    torch::Tensor w_e1, torch::Tensor b_e1,
    torch::Tensor w_e3, torch::Tensor b_e3
){
    CHECK_CUDA(x); CHECK_CUDA(w_sq); CHECK_CUDA(w_e1); CHECK_CUDA(w_e3);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w_sq); CHECK_CONTIGUOUS(w_e1); CHECK_CONTIGUOUS(w_e3);
    CHECK_FLOAT(x); CHECK_FLOAT(w_sq); CHECK_FLOAT(w_e1); CHECK_FLOAT(w_e3);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w_sq.dim() == 4 && w_sq.size(2) == 1 && w_sq.size(3) == 1, "w_sq must be [Csq,Cin,1,1]");
    TORCH_CHECK(w_e1.dim() == 4 && w_e1.size(2) == 1 && w_e1.size(3) == 1, "w_e1 must be [Ce1,Csq,1,1]");
    TORCH_CHECK(w_e3.dim() == 4 && w_e3.size(2) == 3 && w_e3.size(3) == 3, "w_e3 must be [Ce3,Csq,3,3]");

    int N   = (int)x.size(0);
    int Cin = (int)x.size(1);
    int H   = (int)x.size(2);
    int W   = (int)x.size(3);

    int Csq = (int)w_sq.size(0);
    TORCH_CHECK((int)w_sq.size(1) == Cin, "w_sq Cin mismatch");

    int Ce1 = (int)w_e1.size(0);
    TORCH_CHECK((int)w_e1.size(1) == Csq, "w_e1 Csq mismatch");

    int Ce3 = (int)w_e3.size(0);
    TORCH_CHECK((int)w_e3.size(1) == Csq, "w_e3 Csq mismatch");

    const float* b_sq_ptr = nullptr;
    const float* b_e1_ptr = nullptr;
    const float* b_e3_ptr = nullptr;

    if (b_sq.defined() && b_sq.numel() > 0) {
        CHECK_CUDA(b_sq); CHECK_CONTIGUOUS(b_sq); CHECK_FLOAT(b_sq);
        TORCH_CHECK(b_sq.dim() == 1 && (int)b_sq.size(0) == Csq, "b_sq must be [Csq]");
        b_sq_ptr = b_sq.data_ptr<float>();
    }
    if (b_e1.defined() && b_e1.numel() > 0) {
        CHECK_CUDA(b_e1); CHECK_CONTIGUOUS(b_e1); CHECK_FLOAT(b_e1);
        TORCH_CHECK(b_e1.dim() == 1 && (int)b_e1.size(0) == Ce1, "b_e1 must be [Ce1]");
        b_e1_ptr = b_e1.data_ptr<float>();
    }
    if (b_e3.defined() && b_e3.numel() > 0) {
        CHECK_CUDA(b_e3); CHECK_CONTIGUOUS(b_e3); CHECK_FLOAT(b_e3);
        TORCH_CHECK(b_e3.dim() == 1 && (int)b_e3.size(0) == Ce3, "b_e3 must be [Ce3]");
        b_e3_ptr = b_e3.data_ptr<float>();
    }

    int CeT = Ce1 + Ce3;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    auto s = torch::empty({N, Csq, H, W}, x.options());
    auto y = torch::empty({N, CeT, H, W}, x.options());

    // Stage 1: squeeze 1x1 + ReLU
    {
        int HW = H * W;
        int total = N * Csq * HW;
        dim3 block(256, 1, 1);
        dim3 grid((total + (int)block.x - 1) / (int)block.x, 1, 1);

        if (b_sq_ptr) {
            squeeze1x1_relu_kernel<true><<<grid, block, 0, stream>>>(
                x.data_ptr<float>(),
                w_sq.data_ptr<float>(),
                b_sq_ptr,
                s.data_ptr<float>(),
                N, Cin, H, W, Csq
            );
        } else {
            squeeze1x1_relu_kernel<false><<<grid, block, 0, stream>>>(
                x.data_ptr<float>(),
                w_sq.data_ptr<float>(),
                nullptr,
                s.data_ptr<float>(),
                N, Cin, H, W, Csq
            );
        }
        cuda_check_last_error();
    }

    // Stage 2a: expand 1x1 + ReLU
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int OC_BLK = 4;

        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid((W + TILE_W - 1) / TILE_W,
                  (H + TILE_H - 1) / TILE_H,
                  N * ((Ce1 + OC_BLK - 1) / OC_BLK));

        expand1x1_ocblk<TILE_W, TILE_H, OC_BLK><<<grid, block, 0, stream>>>(
            s.data_ptr<float>(),
            w_e1.data_ptr<float>(),
            b_e1_ptr,
            y.data_ptr<float>(),
            N, Csq, H, W, Ce1, CeT
        );
        cuda_check_last_error();
    }

    // Stage 2b: expand 3x3 + ReLU
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int OC_BLK = 2;

        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid((W + TILE_W - 1) / TILE_W,
                  (H + TILE_H - 1) / TILE_H,
                  N * ((Ce3 + OC_BLK - 1) / OC_BLK));

        size_t shmem_bytes = (size_t)(TILE_W + 2) * (size_t)(TILE_H + 2) * sizeof(float);

        expand3x3_ocblk_smem<TILE_W, TILE_H, OC_BLK><<<grid, block, shmem_bytes, stream>>>(
            s.data_ptr<float>(),
            w_e3.data_ptr<float>(),
            b_e3_ptr,
            y.data_ptr<float>(),
            N, Csq, H, W, Ce1, Ce3, CeT
        );
        cuda_check_last_error();
    }

    return y;
}
"""

fire_cpp_source = r"""
torch::Tensor fire_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_sq, torch::Tensor b_sq,
    torch::Tensor w_e1, torch::Tensor b_e1,
    torch::Tensor w_e3, torch::Tensor b_e3
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_squeezenet_fire_twostage_opt",
    cpp_sources=fire_cpp_source,
    cuda_sources=fire_cuda_source,
    functions=["fire_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
)


# -----------------------------
# Fused FireModule using custom op
# -----------------------------
class FireModuleFused(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        Cin = in_channels
        Csq = squeeze_channels
        Ce1 = expand1x1_channels
        Ce3 = expand3x3_channels

        self.w_sq = nn.Parameter(torch.empty(Csq, Cin, 1, 1))
        self.b_sq = nn.Parameter(torch.empty(Csq))

        self.w_e1 = nn.Parameter(torch.empty(Ce1, Csq, 1, 1))
        self.b_e1 = nn.Parameter(torch.empty(Ce1))

        self.w_e3 = nn.Parameter(torch.empty(Ce3, Csq, 3, 3))
        self.b_e3 = nn.Parameter(torch.empty(Ce3))

        for w, b in [(self.w_sq, self.b_sq), (self.w_e1, self.b_e1), (self.w_e3, self.b_e3)]:
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
            fan_in = w.size(1) * w.size(2) * w.size(3)
            bound = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        x = x.contiguous()
        return custom_ops_lib.fire_forward_cuda(
            x,
            self.w_sq.contiguous(), self.b_sq.contiguous(),
            self.w_e1.contiguous(), self.b_e1.contiguous(),
            self.w_e3.contiguous(), self.b_e3.contiguous(),
        )


# -----------------------------
# Full SqueezeNet-like model with fused FireModules
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire2 = FireModuleFused(96, 16, 64, 64)
        self.fire3 = FireModuleFused(128, 16, 64, 64)
        self.fire4 = FireModuleFused(128, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire5 = FireModuleFused(256, 32, 128, 128)
        self.fire6 = FireModuleFused(256, 48, 192, 192)
        self.fire7 = FireModuleFused(384, 48, 192, 192)
        self.fire8 = FireModuleFused(384, 64, 256, 256)
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire9 = FireModuleFused(512, 64, 256, 256)

        self.drop = nn.Dropout(p=0.0)
        self.conv_cls = nn.Conv2d(512, num_classes, kernel_size=1)
        self.relu_cls = nn.ReLU(inplace=True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool4(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool8(x)

        x = self.fire9(x)

        x = self.drop(x)
        x = self.conv_cls(x)
        x = self.relu_cls(x)
        x = self.avg(x)
        return torch.flatten(x, 1)