import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA extension: global_filter_cached_planar_v6
# Improvements over v3:
# - Add lightweight weight packing for the common case f==3:
#     pack W[a,3,C] -> Wpack0[a,C] (float4 holds w0,w1) and Wpack2[a,C] (float2 holds w2)
#   This turns 3 strided weight loads into 1 vector + 1 scalar per channel (better cache line use).
# - Specialized complex multiply for f==3 uses packed weights and unrolls j=0..2 per thread (ILP).
# - Generic fallback kernel unchanged for other f.
# - Avoids shared memory + __syncthreads (do not repeat failed patterns).
# - Keeps pack/unpack vectorization and plan caching.
# ------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cufft.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_CFLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::ComplexFloat, #x " must be complex64 (complex float)")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static inline void cufft_check(cufftResult res, const char* msg) {
    TORCH_CHECK(res == CUFFT_SUCCESS, msg, " (cuFFT error code=", (int)res, ")");
}

struct PlanKey {
    int device;
    int a;
    int b;
    int batch;
    bool operator==(const PlanKey& o) const {
        return device==o.device && a==o.a && b==o.b && batch==o.batch;
    }
};

struct PlanKeyHash {
    std::size_t operator()(PlanKey const& k) const noexcept {
        std::size_t h = (std::size_t)k.device;
        h = h * 1315423911u + (unsigned)k.a;
        h = h * 1315423911u + (unsigned)k.b;
        h = h * 1315423911u + (unsigned)k.batch;
        return h;
    }
};

struct Plans {
    cufftHandle fwd;
    cufftHandle inv;
    bool valid;
};

#include <unordered_map>
#include <mutex>

static std::unordered_map<PlanKey, Plans, PlanKeyHash> g_plans;
static std::mutex g_mutex;

static Plans get_or_create_plans(int device, int a, int b, int batch, cudaStream_t stream) {
    PlanKey key{device, a, b, batch};

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_plans.find(key);
        if (it != g_plans.end()) {
            cufft_check(cufftSetStream(it->second.fwd, stream), "cufftSetStream fwd failed");
            cufft_check(cufftSetStream(it->second.inv, stream), "cufftSetStream inv failed");
            return it->second;
        }
    }

    Plans p;
    p.valid = false;

    cufftHandle plan_fwd;
    cufftHandle plan_inv;

    int n[2] = { a, b };
    int inembed[2]  = { a, b };
    int onembed[2]  = { a, (b/2 + 1) };
    int istride = 1, ostride = 1;
    int idist = a * b;
    int odist = a * (b/2 + 1);

    cufft_check(
        cufftPlanMany(&plan_fwd, 2, n,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_R2C, batch),
        "cufftPlanMany R2C failed"
    );
    cufft_check(
        cufftPlanMany(&plan_inv, 2, n,
                      onembed, ostride, odist,
                      inembed, istride, idist,
                      CUFFT_C2R, batch),
        "cufftPlanMany C2R failed"
    );

    cufft_check(cufftSetStream(plan_fwd, stream), "cufftSetStream fwd failed");
    cufft_check(cufftSetStream(plan_inv, stream), "cufftSetStream inv failed");

    p.fwd = plan_fwd;
    p.inv = plan_inv;
    p.valid = true;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_plans.emplace(key, p);
    }

    return p;
}

static void destroy_all_plans() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto &kv : g_plans) {
        if (kv.second.valid) {
            cufftDestroy(kv.second.fwd);
            cufftDestroy(kv.second.inv);
        }
    }
    g_plans.clear();
}

// pack BHWC -> planar [B*C, a, b]
__global__ void pack_bhwc_to_planar_vec4(
    const float* __restrict__ x_bhwc,
    float* __restrict__ x_planar,
    int B, int a, int b, int C
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t C4 = (int64_t)C / 4;
    int64_t total = (int64_t)B * a * b * C4;
    if (idx >= total) return;

    int64_t t = idx;
    int c4 = (int)(t % C4); t /= C4;
    int j = (int)(t % b);  t /= b;
    int i = (int)(t % a);  t /= a;
    int bidx = (int)t;

    int c = c4 * 4;

    int64_t in_base = (((int64_t)bidx * a + i) * (int64_t)b + j) * (int64_t)C + c;
    float4 v = *reinterpret_cast<const float4*>(x_bhwc + in_base);

    int64_t base_bc = (int64_t)bidx * C + c;
    int64_t out0 = (base_bc + 0) * (int64_t)a * b + (int64_t)i * b + j;
    int64_t out1 = (base_bc + 1) * (int64_t)a * b + (int64_t)i * b + j;
    int64_t out2 = (base_bc + 2) * (int64_t)a * b + (int64_t)i * b + j;
    int64_t out3 = (base_bc + 3) * (int64_t)a * b + (int64_t)i * b + j;

    x_planar[out0] = v.x;
    x_planar[out1] = v.y;
    x_planar[out2] = v.z;
    x_planar[out3] = v.w;
}

__global__ void pack_bhwc_to_planar_scalar(
    const float* __restrict__ x_bhwc,
    float* __restrict__ x_planar,
    int B, int a, int b, int C
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * a * b * C;
    if (idx >= total) return;

    int64_t t = idx;
    int c = (int)(t % C); t /= C;
    int j = (int)(t % b); t /= b;
    int i = (int)(t % a); t /= a;
    int bidx = (int)t;

    int64_t in_off = (((int64_t)bidx * a + i) * (int64_t)b + j) * (int64_t)C + c;
    int64_t bc = (int64_t)bidx * C + c;
    int64_t out_off = bc * (int64_t)a * b + (int64_t)i * b + j;
    x_planar[out_off] = x_bhwc[in_off];
}

// Generic coalesced complex multiply:
// X: [B*C, a, f] bc-major contiguous => contiguous across C for fixed (b,i,j)
// W: [a, f, C] contiguous across C for fixed (i,j)
__global__ void complex_mul_weight_coalesced(
    cufftComplex* __restrict__ X,        // [B*C, a, f]
    const cufftComplex* __restrict__ W,  // [a, f, C]
    int B, int a, int f, int C
) {
    int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int64_t t = (int64_t)blockIdx.y; // 0 .. B*a*f-1
    int j = (int)(t % f);
    t /= f;
    int i = (int)(t % a);
    int bidx = (int)(t / a);

    int64_t bc = (int64_t)bidx * C + c;
    int64_t xoff = bc * (int64_t)a * f + (int64_t)i * f + j;
    int64_t woff = ((int64_t)i * f + j) * (int64_t)C + c;

#if __CUDA_ARCH__ >= 350
    cufftComplex w = __ldg(W + woff);
#else
    cufftComplex w = W[woff];
#endif
    cufftComplex x = X[xoff];
    cufftComplex y;
    y.x = x.x * w.x - x.y * w.y;
    y.y = x.x * w.y + x.y * w.x;
    X[xoff] = y;
}

// Pack weights for f==3:
// W: [a,3,C] complex -> W01: [a,C] float4 (w0 as xy, w1 as zw), W2: [a,C] float2 (w2 as xy)
__global__ void pack_weight_f3(
    const cufftComplex* __restrict__ W, // [a,3,C]
    float4* __restrict__ W01,           // [a,C]
    float2* __restrict__ W2,            // [a,C]
    int a, int C
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)a * (int64_t)C;
    if (idx >= total) return;

    int c = (int)(idx % C);
    int i = (int)(idx / C);

    int64_t base = ((int64_t)i * 3) * (int64_t)C + c; // + j*C
#if __CUDA_ARCH__ >= 350
    cufftComplex w0 = __ldg(W + base + 0LL*(int64_t)C);
    cufftComplex w1 = __ldg(W + base + 1LL*(int64_t)C);
    cufftComplex w2 = __ldg(W + base + 2LL*(int64_t)C);
#else
    cufftComplex w0 = W[base + 0LL*(int64_t)C];
    cufftComplex w1 = W[base + 1LL*(int64_t)C];
    cufftComplex w2 = W[base + 2LL*(int64_t)C];
#endif

    int64_t out = (int64_t)i * (int64_t)C + c;
    W01[out] = make_float4(w0.x, w0.y, w1.x, w1.y);
    W2[out]  = make_float2(w2.x, w2.y);
}

// Specialized multiply for f==3 using packed weights.
// One thread computes (b,i,c) and unrolls j=0..2.
__global__ void complex_mul_f3_packed(
    cufftComplex* __restrict__ X,     // [B*C, a, 3]
    const float4* __restrict__ W01,   // [a,C]
    const float2* __restrict__ W2,    // [a,C]
    int B, int a, int C
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * (int64_t)a * (int64_t)C;
    if (idx >= total) return;

    int c = (int)(idx % C);
    int64_t t = idx / C;
    int i = (int)(t % a);
    int bidx = (int)(t / a);

    const int64_t plane = (int64_t)a * 3;
    const int64_t bc = (int64_t)bidx * (int64_t)C + c;
    const int64_t xbase = bc * plane + (int64_t)i * 3;

    cufftComplex x0 = X[xbase + 0];
    cufftComplex x1 = X[xbase + 1];
    cufftComplex x2 = X[xbase + 2];

    int64_t woff = (int64_t)i * (int64_t)C + c;
#if __CUDA_ARCH__ >= 350
    float4 w01 = __ldg(W01 + woff);
    float2 w2  = __ldg(W2  + woff);
#else
    float4 w01 = W01[woff];
    float2 w2  = W2[woff];
#endif

    // w0 = (w01.x, w01.y), w1 = (w01.z, w01.w), w2 = (w2.x, w2.y)
    cufftComplex y0, y1, y2;
    y0.x = x0.x * w01.x - x0.y * w01.y;  y0.y = x0.x * w01.y + x0.y * w01.x;
    y1.x = x1.x * w01.z - x1.y * w01.w;  y1.y = x1.x * w01.w + x1.y * w01.z;
    y2.x = x2.x * w2.x  - x2.y * w2.y;   y2.y = x2.x * w2.y  + x2.y * w2.x;

    X[xbase + 0] = y0;
    X[xbase + 1] = y1;
    X[xbase + 2] = y2;
}

// Fused: unpack planar -> BHWC and apply scaling by s.
__global__ void unpack_planar_to_bhwc_scale_vec4(
    const float* __restrict__ y_planar,
    float* __restrict__ y_bhwc,
    int B, int a, int b, int C,
    float s
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t C4 = (int64_t)C / 4;
    int64_t total = (int64_t)B * a * b * C4;
    if (idx >= total) return;

    int64_t t = idx;
    int c4 = (int)(t % C4); t /= C4;
    int j = (int)(t % b);  t /= b;
    int i = (int)(t % a);  t /= a;
    int bidx = (int)t;

    int c = c4 * 4;
    int64_t base_bc = (int64_t)bidx * C + c;

    int64_t in0 = (base_bc + 0) * (int64_t)a * b + (int64_t)i * b + j;
    int64_t in1 = (base_bc + 1) * (int64_t)a * b + (int64_t)i * b + j;
    int64_t in2 = (base_bc + 2) * (int64_t)a * b + (int64_t)i * b + j;
    int64_t in3 = (base_bc + 3) * (int64_t)a * b + (int64_t)i * b + j;

    float4 v;
    v.x = y_planar[in0] * s;
    v.y = y_planar[in1] * s;
    v.z = y_planar[in2] * s;
    v.w = y_planar[in3] * s;

    int64_t out_base = (((int64_t)bidx * a + i) * (int64_t)b + j) * (int64_t)C + c;
    *reinterpret_cast<float4*>(y_bhwc + out_base) = v;
}

__global__ void unpack_planar_to_bhwc_scale_scalar(
    const float* __restrict__ y_planar,
    float* __restrict__ y_bhwc,
    int B, int a, int b, int C,
    float s
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * a * b * C;
    if (idx >= total) return;

    int64_t t = idx;
    int c = (int)(t % C); t /= C;
    int j = (int)(t % b); t /= b;
    int i = (int)(t % a); t /= a;
    int bidx = (int)t;

    int64_t bc = (int64_t)bidx * C + c;
    int64_t in_off = bc * (int64_t)a * b + (int64_t)i * b + j;
    int64_t out_off = (((int64_t)bidx * a + i) * (int64_t)b + j) * (int64_t)C + c;
    y_bhwc[out_off] = y_planar[in_off] * s;
}

torch::Tensor global_filter_forward_cuda(torch::Tensor x_bnc, torch::Tensor weight_complex, int64_t a64, int64_t b64) {
    CHECK_INPUT(x_bnc);
    CHECK_INPUT(weight_complex);
    CHECK_FLOAT(x_bnc);
    CHECK_CFLOAT(weight_complex);

    TORCH_CHECK(x_bnc.dim() == 3, "x must be (B,N,C)");
    TORCH_CHECK(weight_complex.dim() == 3, "weight must be (a, b//2+1, C) complex");
    TORCH_CHECK(a64 > 0 && b64 > 0, "Invalid spatial size");
    TORCH_CHECK(x_bnc.size(1) == a64 * b64, "N must equal a*b");

    const int64_t B64 = x_bnc.size(0);
    const int64_t C64 = x_bnc.size(2);
    TORCH_CHECK(B64 <= INT_MAX && C64 <= INT_MAX, "B/C too large");

    const int B = (int)B64;
    const int C = (int)C64;
    const int a = (int)a64;
    const int b = (int)b64;
    const int f = b / 2 + 1;

    TORCH_CHECK(weight_complex.size(0) == a, "weight a mismatch");
    TORCH_CHECK(weight_complex.size(1) == f, "weight f mismatch");
    TORCH_CHECK(weight_complex.size(2) == C64, "weight C mismatch");

    c10::cuda::CUDAGuard device_guard(x_bnc.device());
    const int device = (int)x_bnc.get_device();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto x_bhwc = x_bnc.view({B, a, b, C}).contiguous();

    auto opts_f = torch::TensorOptions().device(x_bnc.device()).dtype(torch::kFloat32);
    auto opts_c = torch::TensorOptions().device(x_bnc.device()).dtype(torch::kComplexFloat);

    int64_t batch = (int64_t)B * (int64_t)C;

    auto x_planar = torch::empty({batch, a, b}, opts_f);
    auto X_freq   = torch::empty({batch, a, f}, opts_c);
    auto y_bhwc   = torch::empty({B, a, b, C}, opts_f);

    const int threads = 256;

    // Pack BHWC -> planar
    {
        bool vec4_ok = (C % 4 == 0) &&
                       (((uintptr_t)x_bhwc.data_ptr<float>() % 16) == 0) &&
                       (((uintptr_t)x_planar.data_ptr<float>() % 16) == 0);
        if (vec4_ok) {
            int64_t total = (int64_t)B * a * b * (C/4);
            int blocks = (int)((total + threads - 1) / threads);
            pack_bhwc_to_planar_vec4<<<blocks, threads, 0, stream>>>(
                x_bhwc.data_ptr<float>(), x_planar.data_ptr<float>(), B, a, b, C
            );
        } else {
            int64_t total = (int64_t)B * a * b * C;
            int blocks = (int)((total + threads - 1) / threads);
            pack_bhwc_to_planar_scalar<<<blocks, threads, 0, stream>>>(
                x_bhwc.data_ptr<float>(), x_planar.data_ptr<float>(), B, a, b, C
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    Plans plans = get_or_create_plans(device, a, b, (int)batch, stream);

    cufft_check(
        cufftExecR2C(plans.fwd,
                     (cufftReal*)x_planar.data_ptr<float>(),
                     (cufftComplex*)X_freq.data_ptr<c10::complex<float>>()),
        "cufftExecR2C failed"
    );

    // Complex multiply
    {
        cufftComplex* X = (cufftComplex*)X_freq.data_ptr<c10::complex<float>>();
        const cufftComplex* W = (const cufftComplex*)weight_complex.data_ptr<c10::complex<float>>();

        if (f == 3) {
            // Pack weights to make weight loads fewer + more contiguous for this kernel.
            // W01: float4[a,C], W2: float2[a,C]
            auto W01_t = torch::empty({a64, C64, 4}, opts_f); // last dim packs float4
            auto W2_t  = torch::empty({a64, C64, 2}, opts_f); // last dim packs float2

            float4* W01 = reinterpret_cast<float4*>(W01_t.data_ptr<float>());
            float2* W2  = reinterpret_cast<float2*>(W2_t.data_ptr<float>());

            int64_t total_pack = (int64_t)a * (int64_t)C;
            int blocks_pack = (int)((total_pack + threads - 1) / threads);
            pack_weight_f3<<<blocks_pack, threads, 0, stream>>>(W, W01, W2, a, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            int64_t total = (int64_t)B * (int64_t)a * (int64_t)C;
            int blocks = (int)((total + threads - 1) / threads);
            complex_mul_f3_packed<<<blocks, threads, 0, stream>>>(X, W01, W2, B, a, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            int mul_threads = (C >= 256) ? 256 : 128;
            int blocks_x = (C + mul_threads - 1) / mul_threads;
            int64_t total_y = (int64_t)B * (int64_t)a * (int64_t)f;
            TORCH_CHECK(total_y <= (int64_t)INT_MAX, "B*a*f too large for grid.y");
            dim3 grid(blocks_x, (unsigned)total_y, 1);
            complex_mul_weight_coalesced<<<grid, mul_threads, 0, stream>>>(X, W, B, a, f, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // iFFT C2R directly into x_planar (reuse buffer)
    cufft_check(
        cufftExecC2R(plans.inv,
                     (cufftComplex*)X_freq.data_ptr<c10::complex<float>>(),
                     (cufftReal*)x_planar.data_ptr<float>()),
        "cufftExecC2R failed"
    );

    // Fused scale+unpack planar -> BHWC
    {
        float s = 1.0f / (float)(a * b); // baseline scaling
        bool vec4_ok = (C % 4 == 0) &&
                       (((uintptr_t)y_bhwc.data_ptr<float>() % 16) == 0) &&
                       (((uintptr_t)x_planar.data_ptr<float>() % 16) == 0);
        if (vec4_ok) {
            int64_t total = (int64_t)B * a * b * (C/4);
            int blocks = (int)((total + threads - 1) / threads);
            unpack_planar_to_bhwc_scale_vec4<<<blocks, threads, 0, stream>>>(
                x_planar.data_ptr<float>(), y_bhwc.data_ptr<float>(), B, a, b, C, s
            );
        } else {
            int64_t total = (int64_t)B * a * b * C;
            int blocks = (int)((total + threads - 1) / threads);
            unpack_planar_to_bhwc_scale_scalar<<<blocks, threads, 0, stream>>>(
                x_planar.data_ptr<float>(), y_bhwc.data_ptr<float>(), B, a, b, C, s
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return y_bhwc.view({B64, a64 * b64, C64});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("global_filter_forward_cuda", &global_filter_forward_cuda, "global_filter_forward_cuda");
    m.def("destroy_all_plans", &destroy_all_plans, "destroy_all_plans");
}
"""

cpp_src = r"""// bindings in CUDA"""

custom_ops_lib = load_inline(
    name="custom_ops_lib",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcufft"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Global Filter module using a custom CUDA op:
    (B,N,C)->(B,a,b,C)->pack to planar [B*C,a,b]->R2C->mul weight->C2R->(scale+unpack)->(B,N,C)
    cuFFT plans are cached across calls.
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.h = int(h)
        self.w = int(w)
        self.dim = int(dim)

    def forward(self, x: torch.Tensor, spatial_size=None) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()

        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.isqrt(N))
            if a * b != N:
                raise RuntimeError("When spatial_size is None, N must be a perfect square")
        else:
            a, b = int(spatial_size[0]), int(spatial_size[1])
            if a * b != N:
                raise RuntimeError("spatial_size a*b must equal N")

        w_complex = torch.view_as_complex(self.complex_weight.contiguous())
        if w_complex.shape != (a, b // 2 + 1, C):
            raise RuntimeError(
                f"Weight shape mismatch: expected ({a}, {b//2+1}, {C}) complex, got {tuple(w_complex.shape)}"
            )

        return custom_ops_lib.global_filter_forward_cuda(x, w_complex.contiguous(), a, b)

    def __del__(self):
        try:
            if torch.cuda.is_available():
                custom_ops_lib.destroy_all_plans()
        except Exception:
            pass