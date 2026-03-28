import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

avg_pool3d_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

static inline int64_t div_up_int64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float2 ldg_f32x2(const float2* p) {
#if __CUDA_ARCH__ >= 350
    float2 v;
    v.x = __ldg(reinterpret_cast<const float*>(&p->x));
    v.y = __ldg(reinterpret_cast<const float*>(&p->y));
    return v;
#else
    return *p;
#endif
}

// ---------------- Generic fallback (any config) ----------------
__global__ __launch_bounds__(256, 2)
void avg_pool3d_forward_linear_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)outD * (int64_t)outH * (int64_t)outW;
    if (idx >= total) return;

    int ow = (int)(idx % outW);
    int64_t t = idx / outW;
    int oh = (int)(t % outH);
    t /= outH;
    int od = (int)(t % outD);
    t /= outD;
    int c = (int)(t % C);
    int n = (int)(t / C);

    int dstart = od * sD - pD;
    int hstart = oh * sH - pH;
    int wstart = ow * sW - pW;

    int dend = dstart + kD;
    int hend = hstart + kH;
    int wend = wstart + kW;

    int d0 = dstart < 0 ? 0 : dstart;
    int h0 = hstart < 0 ? 0 : hstart;
    int w0 = wstart < 0 ? 0 : wstart;

    int d1 = dend > D ? D : dend;
    int h1 = hend > H ? H : hend;
    int w1 = wend > W ? W : wend;

    float sum = 0.0f;
    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t base_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)D * HW;

    for (int id = d0; id < d1; ++id) {
        int64_t base_d = base_nc + (int64_t)id * HW;
        for (int ih = h0; ih < h1; ++ih) {
            int64_t row_base = base_d + (int64_t)ih * (int64_t)W;

            int iw = w0;
            int len = w1 - w0;

            if ((((row_base + (int64_t)iw) & 3LL) == 0) && (len >= 4)) {
                int n4 = len >> 2;
                const float4* p4 = reinterpret_cast<const float4*>(x + row_base + (int64_t)iw);
                #pragma unroll 1
                for (int j = 0; j < n4; ++j) {
                    float4 v = p4[j];
                    sum += (v.x + v.y) + (v.z + v.w);
                }
                iw += (n4 << 2);
            }

            int rem = w1 - iw;
            if ((((row_base + (int64_t)iw) & 1LL) == 0) && (rem >= 2)) {
                int n2 = rem >> 1;
                const float2* p2 = reinterpret_cast<const float2*>(x + row_base + (int64_t)iw);
                #pragma unroll 1
                for (int j = 0; j < n2; ++j) {
                    float2 v = p2[j];
                    sum += v.x + v.y;
                }
                iw += (n2 << 1);
            }

            for (; iw < w1; ++iw) sum += ldg_f32(x + row_base + (int64_t)iw);
        }
    }

    float denom = (float)(kD * kH * kW); // count_include_pad=True
    y[idx] = sum / denom;
}

// ---------------- Specialized k=3,s=2,p=1 ----------------
static inline int interior_o_min() { return 1; }
static inline int interior_o_max(int X) { return (X - 2) >> 1; } // floor((X-2)/2)

__global__ __launch_bounds__(128, 3)
void avg_pool3d_forward_k3s2p1_interior_3d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int od_min, int od_max,
    int oh_min, int oh_max,
    int ow_min, int ow_max
) {
    // grid.x tiles ow, grid.y is oh, grid.z packs (n,c,od)
    int ow = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x + ow_min;
    if (ow > ow_max) return;

    int oh = (int)blockIdx.y + oh_min;

    int od_range = od_max - od_min + 1;
    int packed = (int)blockIdx.z;
    int nc = packed / od_range;
    int od = (packed - nc * od_range) + od_min;
    int n = nc / C;
    int c = nc - n * C;

    // in-bounds starts
    int dstart = od * 2 - 1;
    int hstart = oh * 2 - 1;
    int wstart = ow * 2 - 1;

    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t base_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)D * HW;
    int64_t b0 = base_nc + (int64_t)dstart * HW + (int64_t)hstart * (int64_t)W + (int64_t)wstart;

    float sum = 0.0f;

    // fixed 3x3x3, unrolled. Keep pointer arithmetic simple to reduce live ranges.
    #pragma unroll
    for (int dz = 0; dz < 3; ++dz) {
        int64_t bd = b0 + (int64_t)dz * HW;
        #pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            int64_t br = bd + (int64_t)dy * (int64_t)W;
            const float* p = x + br;

            // load 3 floats; attempt float2 if 8B aligned
            if ((br & 1LL) == 0LL) {
                float2 v2 = ldg_f32x2(reinterpret_cast<const float2*>(p));
                sum += v2.x + v2.y;
                sum += ldg_f32(p + 2);
            } else {
                sum += ldg_f32(p + 0);
                sum += ldg_f32(p + 1);
                sum += ldg_f32(p + 2);
            }
        }
    }

    int64_t out_idx =
        (((((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)outD + (int64_t)od) * (int64_t)outH + (int64_t)oh) * (int64_t)outW + (int64_t)ow);
    y[out_idx] = sum * (1.0f / 27.0f);
}

// Border-only kernel over an explicit slab region [od0..od1], [oh0..oh1], [ow0..ow1]
__global__ __launch_bounds__(256, 2)
void avg_pool3d_forward_k3s2p1_border_slab_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int od0, int od1,
    int oh0, int oh1,
    int ow0, int ow1
) {
    int ow = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x + ow0;
    if (ow > ow1) return;

    int oh = (int)blockIdx.y + oh0;
    int packed = (int)blockIdx.z; // packs (n,c,od in slab)
    int od_range = od1 - od0 + 1;
    int nc = packed / od_range;
    int od = (packed - nc * od_range) + od0;
    int n = nc / C;
    int c = nc - n * C;

    int dstart = od * 2 - 1;
    int hstart = oh * 2 - 1;
    int wstart = ow * 2 - 1;

    int d0i = dstart < 0 ? 0 : dstart;
    int h0i = hstart < 0 ? 0 : hstart;
    int w0i = wstart < 0 ? 0 : wstart;

    int d1i = dstart + 3; if (d1i > D) d1i = D;
    int h1i = hstart + 3; if (h1i > H) h1i = H;
    int w1i = wstart + 3; if (w1i > W) w1i = W;

    float sum = 0.0f;
    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t base_nc = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)D * HW;

    for (int id = d0i; id < d1i; ++id) {
        int64_t base_d = base_nc + (int64_t)id * HW;
        for (int ih = h0i; ih < h1i; ++ih) {
            int64_t row_base = base_d + (int64_t)ih * (int64_t)W;
            int len = w1i - w0i;
            int iw = w0i;

            if (len == 3) {
                if ((row_base + (int64_t)iw & 1LL) == 0LL) {
                    float2 v2 = ldg_f32x2(reinterpret_cast<const float2*>(x + row_base + (int64_t)iw));
                    sum += v2.x + v2.y;
                    sum += ldg_f32(x + row_base + (int64_t)(iw + 2));
                } else {
                    sum += ldg_f32(x + row_base + (int64_t)iw);
                    sum += ldg_f32(x + row_base + (int64_t)(iw + 1));
                    sum += ldg_f32(x + row_base + (int64_t)(iw + 2));
                }
            } else if (len == 2) {
                if ((row_base + (int64_t)iw & 1LL) == 0LL) {
                    float2 v2 = ldg_f32x2(reinterpret_cast<const float2*>(x + row_base + (int64_t)iw));
                    sum += v2.x + v2.y;
                } else {
                    sum += ldg_f32(x + row_base + (int64_t)iw);
                    sum += ldg_f32(x + row_base + (int64_t)(iw + 1));
                }
            } else if (len == 1) {
                sum += ldg_f32(x + row_base + (int64_t)iw);
            }
        }
    }

    int64_t out_idx =
        (((((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)outD + (int64_t)od) * (int64_t)outH + (int64_t)oh) * (int64_t)outW + (int64_t)ow);
    y[out_idx] = sum * (1.0f / 27.0f);
}

static void launch_border_slab(
    const float* x, float* y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int od0, int od1,
    int oh0, int oh1,
    int ow0, int ow1,
    cudaStream_t stream
) {
    if (od0 > od1 || oh0 > oh1 || ow0 > ow1) return;

    const int threads = 256;
    int ow_count = ow1 - ow0 + 1;

    dim3 block(threads, 1, 1);
    dim3 grid((unsigned)div_up_int64(ow_count, threads),
              (unsigned)(oh1 - oh0 + 1),
              (unsigned)((int64_t)N * (int64_t)C * (int64_t)(od1 - od0 + 1)));

    avg_pool3d_forward_k3s2p1_border_slab_kernel<<<grid, block, 0, stream>>>(
        x, y, N, C, D, H, W, outD, outH, outW,
        od0, od1, oh0, oh1, ow0, ow1
    );
}

static void launch_k3s2p1_split(
    const float* x, float* y,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    cudaStream_t stream
) {
    int od_min = interior_o_min();
    int oh_min = interior_o_min();
    int ow_min = interior_o_min();
    int od_max = interior_o_max(D);
    int oh_max = interior_o_max(H);
    int ow_max = interior_o_max(W);

    bool has_interior = (od_min <= od_max) && (oh_min <= oh_max) && (ow_min <= ow_max) &&
                        (od_max < outD) && (oh_max < outH) && (ow_max < outW);

    if (has_interior) {
        // interior launch (3D grid)
        {
            const int threads = 128;
            int ow_count = (ow_max - ow_min + 1);
            dim3 block(threads, 1, 1);
            dim3 grid((unsigned)div_up_int64(ow_count, threads),
                      (unsigned)(oh_max - oh_min + 1),
                      (unsigned)((int64_t)N * (int64_t)C * (int64_t)(od_max - od_min + 1)));

            avg_pool3d_forward_k3s2p1_interior_3d_kernel<<<grid, block, 0, stream>>>(
                x, y, N, C, D, H, W, outD, outH, outW,
                od_min, od_max, oh_min, oh_max, ow_min, ow_max
            );
        }

        // True border-only launches: 6 slabs (some may be empty).
        // 1) D low: od in [0, od_min-1], all oh/ow
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           0, od_min - 1,
                           0, outH - 1,
                           0, outW - 1,
                           stream);
        // 2) D high: od in [od_max+1, outD-1], all oh/ow
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           od_max + 1, outD - 1,
                           0, outH - 1,
                           0, outW - 1,
                           stream);

        // For remaining slabs, restrict od to interior od range.
        // 3) H low
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           od_min, od_max,
                           0, oh_min - 1,
                           0, outW - 1,
                           stream);
        // 4) H high
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           od_min, od_max,
                           oh_max + 1, outH - 1,
                           0, outW - 1,
                           stream);

        // Restrict od+oh to interior ranges for W slabs
        // 5) W low
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           od_min, od_max,
                           oh_min, oh_max,
                           0, ow_min - 1,
                           stream);
        // 6) W high
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           od_min, od_max,
                           oh_min, oh_max,
                           ow_max + 1, outW - 1,
                           stream);
    } else {
        // No interior exists: treat everything as border, single slab over full output
        launch_border_slab(x, y, N, C, D, H, W, outD, outH, outW,
                           0, outD - 1, 0, outH - 1, 0, outW - 1, stream);
    }
}

torch::Tensor avg_pool3d_forward_cuda(
    torch::Tensor x,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    TORCH_CHECK(x.is_cuda(), "avg_pool3d_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "avg_pool3d_forward_cuda: only float32 supported");
    TORCH_CHECK(x.dim() == 5, "avg_pool3d_forward_cuda: expected input of shape (N, C, D, H, W)");
    TORCH_CHECK(kD > 0 && kH > 0 && kW > 0, "kernel sizes must be > 0");
    TORCH_CHECK(sD > 0 && sH > 0 && sW > 0, "strides must be > 0");
    TORCH_CHECK(pD >= 0 && pH >= 0 && pW >= 0, "paddings must be >= 0");

    auto x_contig = x.contiguous();

    const int64_t N64 = x_contig.size(0);
    const int64_t C64 = x_contig.size(1);
    const int64_t D64 = x_contig.size(2);
    const int64_t H64 = x_contig.size(3);
    const int64_t W64 = x_contig.size(4);

    const int64_t outD64 = (D64 + 2 * pD - kD) / sD + 1;
    const int64_t outH64 = (H64 + 2 * pH - kH) / sH + 1;
    const int64_t outW64 = (W64 + 2 * pW - kW) / sW + 1;
    TORCH_CHECK(outD64 >= 0 && outH64 >= 0 && outW64 >= 0, "avg_pool3d_forward_cuda: computed output size is negative");

    auto y = torch::empty({N64, C64, outD64, outH64, outW64}, x_contig.options());

    TORCH_CHECK(N64 <= INT_MAX && C64 <= INT_MAX && D64 <= INT_MAX && H64 <= INT_MAX && W64 <= INT_MAX, "dims too large");
    TORCH_CHECK(outD64 <= INT_MAX && outH64 <= INT_MAX && outW64 <= INT_MAX, "out dims too large");

    const int N = (int)N64;
    const int C = (int)C64;
    const int D = (int)D64;
    const int H = (int)H64;
    const int W = (int)W64;
    const int outD = (int)outD64;
    const int outH = (int)outH64;
    const int outW = (int)outW64;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const bool use_k3s2p1 =
        (kD == 3 && kH == 3 && kW == 3 &&
         sD == 2 && sH == 2 && sW == 2 &&
         pD == 1 && pH == 1 && pW == 1);

    if (use_k3s2p1) {
        launch_k3s2p1_split(
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, D, H, W,
            outD, outH, outW,
            stream
        );
    } else {
        const int threads = 256;
        const int64_t total = (int64_t)N * (int64_t)C * (int64_t)outD * (int64_t)outH * (int64_t)outW;
        const int blocks = (int)div_up_int64(total, threads);

        avg_pool3d_forward_linear_kernel<<<blocks, threads, 0, stream>>>(
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, D, H, W,
            outD, outH, outW,
            (int)kD, (int)kH, (int)kW,
            (int)sD, (int)sH, (int)sW,
            (int)pD, (int)pH, (int)pW
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avg_pool3d kernel launch failed: ", cudaGetErrorString(err));
    return y;
}
"""

avg_pool3d_cpp_src = r"""
#include <torch/extension.h>

torch::Tensor avg_pool3d_forward_cuda(
    torch::Tensor x,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_avgpool3d_v6_true_border_slabs_lb128",
    cpp_sources=avg_pool3d_cpp_src,
    cuda_sources=avg_pool3d_cuda_src,
    functions=["avg_pool3d_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--ptxas-options=-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Replacement model using an optimized custom CUDA kernel for AvgPool3d forward.
    Fast path supports CUDA float32 NCDHW; falls back otherwise.
    Matches nn.AvgPool3d defaults: ceil_mode=False, count_include_pad=True, divisor_override=None.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        k = int(kernel_size)
        s = k if stride is None else int(stride)
        p = int(padding)

        self.kD = k
        self.kH = k
        self.kW = k
        self.sD = s
        self.sH = s
        self.sW = s
        self.pD = p
        self.pH = p
        self.pW = p

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 5):
            return F.avg_pool3d(
                x,
                kernel_size=(self.kD, self.kH, self.kW),
                stride=(self.sD, self.sH, self.sW),
                padding=(self.pD, self.pH, self.pW),
                ceil_mode=False,
                count_include_pad=True,
                divisor_override=None,
            )

        return self.custom_ops.avg_pool3d_forward_cuda(
            x,
            self.kD, self.kH, self.kW,
            self.sD, self.sH, self.sW,
            self.pD, self.pH, self.pW,
        )