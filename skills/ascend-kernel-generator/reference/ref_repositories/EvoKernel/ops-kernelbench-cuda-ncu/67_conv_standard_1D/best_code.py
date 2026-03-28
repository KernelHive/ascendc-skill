import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv_standard1d (optimized with packed weights) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

// Host-side alignment helper (avoid device/host mismatch from prior failed attempt).
static inline bool is_aligned_16_host(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xFULL) == 0);
}

// Packed weight layout (Cout-major blocks of 4):
// wp: [Cout4, Cin, K, 4] contiguous in last dim
// where Cout4 = ceil(Cout/4)
// Access: wp[(((co4 * Cin + cin) * K + k) * 4 + lane)]
__global__ void conv1d_k3s1d1_f32_packed_coutvec8_kernel(
    const float* __restrict__ x,     // [N, Cin, Lin]
    const float* __restrict__ wp,    // [Cout4, Cin, 3, 4] packed
    const float* __restrict__ b,     // [Cout] or nullptr
    float* __restrict__ y,           // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout, int Lout,
    int padding,
    bool has_bias
) {
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Lout;
    if (t >= total) return;

    int lout = t % Lout;
    int n = t / Lout;

    // each blockIdx.y selects a CoutVec8 tile (8 consecutive output channels)
    int cout0 = (int)blockIdx.y * 8;
    if (cout0 >= Cout) return;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

    if (has_bias) {
        if (cout0 + 0 < Cout) acc0 = ldg_f32(b + (cout0 + 0));
        if (cout0 + 1 < Cout) acc1 = ldg_f32(b + (cout0 + 1));
        if (cout0 + 2 < Cout) acc2 = ldg_f32(b + (cout0 + 2));
        if (cout0 + 3 < Cout) acc3 = ldg_f32(b + (cout0 + 3));
        if (cout0 + 4 < Cout) acc4 = ldg_f32(b + (cout0 + 4));
        if (cout0 + 5 < Cout) acc5 = ldg_f32(b + (cout0 + 5));
        if (cout0 + 6 < Cout) acc6 = ldg_f32(b + (cout0 + 6));
        if (cout0 + 7 < Cout) acc7 = ldg_f32(b + (cout0 + 7));
    }

    int base_in = lout - padding; // stride=1, dilation=1

    const float* __restrict__ x_n = x + (size_t)n * (size_t)Cin * (size_t)Lin;
    float* __restrict__ y_n = y + (size_t)n * (size_t)Cout * (size_t)Lout;

    int co4a = cout0 >> 2;        // first group of 4
    int co4b = (cout0 + 4) >> 2;  // second group of 4

    #pragma unroll 1
    for (int cin = 0; cin < Cin; ++cin) {
        const float* __restrict__ x_ptr = x_n + (size_t)cin * (size_t)Lin;

        float x0 = 0.f, x1 = 0.f, x2 = 0.f;
        int lin0 = base_in;
        int lin1 = base_in + 1;
        int lin2 = base_in + 2;
        if ((unsigned)lin0 < (unsigned)Lin) x0 = ldg_f32(x_ptr + lin0);
        if ((unsigned)lin1 < (unsigned)Lin) x1 = ldg_f32(x_ptr + lin1);
        if ((unsigned)lin2 < (unsigned)Lin) x2 = ldg_f32(x_ptr + lin2);

        // Load packed weights as float4 for each k
        const float* __restrict__ wbase_a = wp + (((size_t)co4a * (size_t)Cin + (size_t)cin) * 3) * 4;
        const float* __restrict__ wbase_b = wp + (((size_t)co4b * (size_t)Cin + (size_t)cin) * 3) * 4;

        float4 wa0 = *reinterpret_cast<const float4*>(wbase_a + 0 * 4);
        float4 wa1 = *reinterpret_cast<const float4*>(wbase_a + 1 * 4);
        float4 wa2 = *reinterpret_cast<const float4*>(wbase_a + 2 * 4);

        float4 wb0 = *reinterpret_cast<const float4*>(wbase_b + 0 * 4);
        float4 wb1 = *reinterpret_cast<const float4*>(wbase_b + 1 * 4);
        float4 wb2 = *reinterpret_cast<const float4*>(wbase_b + 2 * 4);

        // First 4 output channels
        acc0 = fmaf(x0, wa0.x, acc0); acc0 = fmaf(x1, wa1.x, acc0); acc0 = fmaf(x2, wa2.x, acc0);
        acc1 = fmaf(x0, wa0.y, acc1); acc1 = fmaf(x1, wa1.y, acc1); acc1 = fmaf(x2, wa2.y, acc1);
        acc2 = fmaf(x0, wa0.z, acc2); acc2 = fmaf(x1, wa1.z, acc2); acc2 = fmaf(x2, wa2.z, acc2);
        acc3 = fmaf(x0, wa0.w, acc3); acc3 = fmaf(x1, wa1.w, acc3); acc3 = fmaf(x2, wa2.w, acc3);

        // Next 4 output channels
        acc4 = fmaf(x0, wb0.x, acc4); acc4 = fmaf(x1, wb1.x, acc4); acc4 = fmaf(x2, wb2.x, acc4);
        acc5 = fmaf(x0, wb0.y, acc5); acc5 = fmaf(x1, wb1.y, acc5); acc5 = fmaf(x2, wb2.y, acc5);
        acc6 = fmaf(x0, wb0.z, acc6); acc6 = fmaf(x1, wb1.z, acc6); acc6 = fmaf(x2, wb2.z, acc6);
        acc7 = fmaf(x0, wb0.w, acc7); acc7 = fmaf(x1, wb1.w, acc7); acc7 = fmaf(x2, wb2.w, acc7);
    }

    // Store (scalar stores; contiguous over lout for each cout across threads)
    if (cout0 + 0 < Cout) y_n[(size_t)(cout0 + 0) * (size_t)Lout + (size_t)lout] = acc0;
    if (cout0 + 1 < Cout) y_n[(size_t)(cout0 + 1) * (size_t)Lout + (size_t)lout] = acc1;
    if (cout0 + 2 < Cout) y_n[(size_t)(cout0 + 2) * (size_t)Lout + (size_t)lout] = acc2;
    if (cout0 + 3 < Cout) y_n[(size_t)(cout0 + 3) * (size_t)Lout + (size_t)lout] = acc3;
    if (cout0 + 4 < Cout) y_n[(size_t)(cout0 + 4) * (size_t)Lout + (size_t)lout] = acc4;
    if (cout0 + 5 < Cout) y_n[(size_t)(cout0 + 5) * (size_t)Lout + (size_t)lout] = acc5;
    if (cout0 + 6 < Cout) y_n[(size_t)(cout0 + 6) * (size_t)Lout + (size_t)lout] = acc6;
    if (cout0 + 7 < Cout) y_n[(size_t)(cout0 + 7) * (size_t)Lout + (size_t)lout] = acc7;
}

template<int COUT_VEC>
__global__ void conv1d_general_f32_coutvec_kernel(
    const float* __restrict__ x,   // [N, Cin, Lin]
    const float* __restrict__ w,   // [Cout, Cin, K]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N, Cout, Lout]
    int N, int Cin, int Lin,
    int Cout, int K,
    int stride, int padding, int dilation,
    int Lout,
    bool has_bias
) {
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Lout;
    if (t >= total) return;

    int lout = t % Lout;
    int n = t / Lout;

    int cout0 = (int)blockIdx.y * COUT_VEC;
    if (cout0 >= Cout) return;

    float acc[COUT_VEC];
    #pragma unroll
    for (int i = 0; i < COUT_VEC; ++i) {
        int c = cout0 + i;
        float v = 0.0f;
        if (has_bias && c < Cout) v = ldg_f32(b + c);
        acc[i] = v;
    }

    int base_in = lout * stride - padding;
    const float* __restrict__ x_n = x + (size_t)n * (size_t)Cin * (size_t)Lin;
    float* __restrict__ y_n = y + (size_t)n * (size_t)Cout * (size_t)Lout;

    #pragma unroll 1
    for (int cin = 0; cin < Cin; ++cin) {
        const float* __restrict__ x_ptr = x_n + (size_t)cin * (size_t)Lin;

        #pragma unroll 1
        for (int k = 0; k < K; ++k) {
            int lin = base_in + k * dilation;
            if ((unsigned)lin < (unsigned)Lin) {
                float xv = ldg_f32(x_ptr + lin);
                #pragma unroll
                for (int i = 0; i < COUT_VEC; ++i) {
                    int cout = cout0 + i;
                    if (cout >= Cout) continue;
                    const float* __restrict__ w_ptr = w + ((size_t)cout * (size_t)Cin + (size_t)cin) * (size_t)K;
                    acc[i] = fmaf(xv, ldg_f32(w_ptr + k), acc[i]);
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < COUT_VEC; ++i) {
        int cout = cout0 + i;
        if (cout < Cout) {
            y_n[(size_t)cout * (size_t)Lout + (size_t)lout] = acc[i];
        }
    }
}

torch::Tensor conv_standard1d_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor wp, // optional packed weights for K=3,s=1,d=1
    torch::Tensor b,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N,Cin,Lin)");
    TORCH_CHECK(w.dim() == 3, "w must be 3D (Cout,Cin,K)");
    TORCH_CHECK(x.size(1) == w.size(1), "Cin mismatch between x and w");

    bool has_bias = false;
    if (b.defined() && b.numel() > 0) {
        TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
        TORCH_CHECK(b.dim() == 1, "b must be 1D (Cout)");
        TORCH_CHECK(b.size(0) == w.size(0), "Cout mismatch between b and w");
        has_bias = true;
    }

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();
    if (has_bias && !b.is_contiguous()) b = b.contiguous();
    if (wp.defined() && wp.numel() > 0 && !wp.is_contiguous()) wp = wp.contiguous();

    const int N = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int Lin = (int)x.size(2);
    const int Cout = (int)w.size(0);
    const int K = (int)w.size(2);

    const int s = (int)stride;
    const int p = (int)padding;
    const int d = (int)dilation;

    TORCH_CHECK(s > 0, "stride must be > 0");
    TORCH_CHECK(d > 0, "dilation must be > 0");
    TORCH_CHECK(K > 0, "kernel size must be > 0");

    int Lout = (Lin + 2 * p - d * (K - 1) - 1) / s + 1;
    TORCH_CHECK(Lout > 0, "Computed Lout <= 0; check parameters");

    auto y = torch::empty({N, Cout, Lout}, x.options());

    const float* b_ptr = has_bias ? (const float*)b.data_ptr<float>() : nullptr;

    // Prefer packed K=3, s=1, d=1 path if wp is provided and aligned for float4 loads.
    bool use_packed = (K == 3 && s == 1 && d == 1 && wp.defined() && wp.numel() > 0 && wp.is_cuda() && wp.dtype() == torch::kFloat32);
    if (use_packed) {
        // Expect wp shape: [Cout4, Cin, 3, 4]
        TORCH_CHECK(wp.dim() == 4, "wp must be 4D [Cout4,Cin,3,4]");
        TORCH_CHECK((int)wp.size(1) == Cin, "wp Cin mismatch");
        TORCH_CHECK((int)wp.size(2) == 3, "wp K must be 3");
        TORCH_CHECK((int)wp.size(3) == 4, "wp last dim must be 4");
        // float4 alignment
        TORCH_CHECK(is_aligned_16_host(wp.data_ptr()), "wp must be 16-byte aligned");

        const int threads = 256;
        const int total = N * Lout;
        const int blocks_x = (total + threads - 1) / threads;
        const int blocks_y = (Cout + 8 - 1) / 8;

        dim3 grid(blocks_x, blocks_y, 1);
        dim3 block(threads, 1, 1);

        conv1d_k3s1d1_f32_packed_coutvec8_kernel<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)wp.data_ptr<float>(),
            b_ptr,
            (float*)y.data_ptr<float>(),
            N, Cin, Lin,
            Cout, Lout,
            p,
            has_bias
        );
        return y;
    }

    // General fallback: CoutVec=4
    constexpr int COUTV = 4;
    const int threads = 256;
    const int total = N * Lout;
    const int blocks_x = (total + threads - 1) / threads;
    const int blocks_y = (Cout + COUTV - 1) / COUTV;

    dim3 grid(blocks_x, blocks_y, 1);
    dim3 block(threads, 1, 1);

    conv1d_general_f32_coutvec_kernel<COUTV><<<grid, block>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)w.data_ptr<float>(),
        b_ptr,
        (float*)y.data_ptr<float>(),
        N, Cin, Lin,
        Cout, K,
        s, p, d,
        Lout,
        has_bias
    );

    return y;
}
"""

cpp_src = r"""
torch::Tensor conv_standard1d_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor wp,
    torch::Tensor b,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_standard1d_packedw",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv_standard1d_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    1D convolution using a custom CUDA kernel (float32, groups=1 fast path),
    with optional packed-weight fast path for K=3, stride=1, dilation=1.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.custom_ops_lib = custom_ops_lib

        # Cache for packed weights (recomputed if weight storage changes)
        self._wp_cache = None
        self._wp_cache_key = None  # (data_ptr, device, dtype, shape, version)

    @staticmethod
    def _pack_w_cout4(w: torch.Tensor) -> torch.Tensor:
        # w: [Cout, Cin, K] float32 contiguous CUDA
        # returns wp: [Cout4, Cin, K, 4] with zero padding on Cout tail
        Cout, Cin, K = w.shape
        Cout4 = (Cout + 3) // 4
        wp = w.new_zeros((Cout4, Cin, K, 4), dtype=torch.float32)
        # reshape into blocks of 4 output channels
        w_pad = w.new_zeros((Cout4 * 4, Cin, K), dtype=torch.float32)
        w_pad[:Cout].copy_(w)
        wp.copy_(w_pad.view(Cout4, 4, Cin, K).permute(0, 2, 3, 1).contiguous())
        return wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv1d.groups != 1:
            return self.conv1d(x)

        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous().to(dtype=torch.float32)

        w = self.conv1d.weight.contiguous().to(device=x.device, dtype=torch.float32)
        b = self.conv1d.bias
        if b is None:
            b = x.new_empty((0,), dtype=torch.float32)
        else:
            b = b.contiguous().to(device=x.device, dtype=torch.float32)

        s = int(self.conv1d.stride[0])
        p = int(self.conv1d.padding[0])
        d = int(self.conv1d.dilation[0])
        K = int(w.shape[2])

        # Build packed weights only for the hot path.
        wp = x.new_empty((0,), dtype=torch.float32)
        if K == 3 and s == 1 and d == 1:
            # Cache by underlying storage pointer + version to avoid repacking every forward.
            # _version increments on in-place updates; replacing the Parameter changes data_ptr.
            key = (int(w.data_ptr()), str(w.device), str(w.dtype), tuple(w.shape), int(getattr(w, "_version", 0)))
            if self._wp_cache is None or self._wp_cache_key != key:
                self._wp_cache = self._pack_w_cout4(w)
                self._wp_cache_key = key
            wp = self._wp_cache

        return self.custom_ops_lib.conv_standard1d_cuda(
            x, w, wp, b, s, p, d
        )