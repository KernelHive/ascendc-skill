import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }

#if __CUDA_ARCH__ >= 350
#define LDG(p) __ldg(p)
#else
#define LDG(p) (*(p))
#endif

#include <cuda_fp16.h>

// ---- Model constants ----
constexpr int C1_OUT = 6;
constexpr int C1_IN  = 1;
constexpr int K1 = 5;

constexpr int C2_OUT = 16;
constexpr int C2_IN  = 6;
constexpr int K2 = 5;

constexpr int H0 = 32, W0 = 32;
constexpr int H1 = 28, W1 = 28;
constexpr int H1P = 14, W1P = 14;
constexpr int H2 = 10, W2 = 10;
constexpr int H2P = 5, W2P = 5;

constexpr int FLAT = C2_OUT * H2P * W2P; // 400
constexpr int FC1_OUT = 120;
constexpr int FC2_OUT = 84;

// ---- Constant memory for small conv weights/biases ----
__constant__ float c_w1[C1_OUT * C1_IN * K1 * K1]; // 150
__constant__ float c_b1[C1_OUT];                   // 6
__constant__ float c_w2[C2_OUT * C2_IN * K2 * K2]; // 2400
__constant__ float c_b2[C2_OUT];                   // 16

static uintptr_t g_last_w1 = 0;
static uintptr_t g_last_b1 = 0;
static uintptr_t g_last_w2 = 0;
static uintptr_t g_last_b2 = 0;

static inline void maybe_copy_conv_to_const(torch::Tensor w1, torch::Tensor b1,
                                            torch::Tensor w2, torch::Tensor b2) {
    uintptr_t pw1 = (uintptr_t)w1.data_ptr<float>();
    uintptr_t pb1 = (uintptr_t)b1.data_ptr<float>();
    uintptr_t pw2 = (uintptr_t)w2.data_ptr<float>();
    uintptr_t pb2 = (uintptr_t)b2.data_ptr<float>();
    if (pw1 != g_last_w1 || pb1 != g_last_b1 || pw2 != g_last_w2 || pb2 != g_last_b2) {
        cudaMemcpyToSymbol(c_w1, w1.data_ptr<float>(), sizeof(float) * C1_OUT * C1_IN * K1 * K1);
        cudaMemcpyToSymbol(c_b1, b1.data_ptr<float>(), sizeof(float) * C1_OUT);
        cudaMemcpyToSymbol(c_w2, w2.data_ptr<float>(), sizeof(float) * C2_OUT * C2_IN * K2 * K2);
        cudaMemcpyToSymbol(c_b2, b2.data_ptr<float>(), sizeof(float) * C2_OUT);
        g_last_w1 = pw1; g_last_b1 = pb1; g_last_w2 = pw2; g_last_b2 = pb2;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// Convert float->half kernel
__global__ void f32_to_f16_kernel(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = __float2half_rn(in[i]);
}

// Fused inference kernel.
// Block per sample; 128 threads to reduce wasted warps and improve occupancy.
__global__ __launch_bounds__(128, 3) void lenet5_infer_kernel_opt5(
    const float* __restrict__ x,          // [N,1,32,32]
    const half*  __restrict__ w3h,        // [120,400] half
    const float* __restrict__ b3,         // [120] float
    const half*  __restrict__ w4h,        // [84,120] half
    const float* __restrict__ b4,         // [84] float
    const half*  __restrict__ w5h,        // [Cout,84] half
    const float* __restrict__ b5,         // [Cout] float
    float* __restrict__ y,                // [N,Cout]
    int N,
    int C_out
) {
    int n = (int)blockIdx.x;
    if (n >= N) return;

    extern __shared__ unsigned char smem_raw[];
    float* xsh = reinterpret_cast<float*>(smem_raw);                  // 32*32 = 1024 floats
    half*  c1p_h = reinterpret_cast<half*>(xsh + H0*W0);              // 6*14*14 = 1176 half
    float* c2p = reinterpret_cast<float*>(c1p_h + (C1_OUT*H1P*W1P));  // 400 float
    float* fc1 = c2p + (C2_OUT*H2P*W2P);                              // 120 float
    float* fc2 = fc1 + FC1_OUT;                                       // 84 float

    // ---- stage input x into shared (coalesced) ----
    const float* xg = x + n * (H0 * W0);
    for (int i = (int)threadIdx.x; i < H0*W0; i += (int)blockDim.x) {
        xsh[i] = LDG(xg + i);
    }
    __syncthreads();

    // ---- conv1 + relu + maxpool2 into half shared ----
    for (int idx = (int)threadIdx.x; idx < C1_OUT * H1P * W1P; idx += (int)blockDim.x) {
        int pw = idx % W1P;
        int t1 = idx / W1P;
        int ph = t1 % H1P;
        int oc = t1 / H1P;

        float m = 0.0f;
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            int oh = ph * 2 + dy;
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                int ow = pw * 2 + dx;
                float acc = c_b1[oc];
                int w_base = (oc * K1) * K1;

                #pragma unroll
                for (int ky = 0; ky < K1; ++ky) {
                    int iy = oh + ky;
                    int x_row = iy * W0;
                    #pragma unroll
                    for (int kx = 0; kx < K1; ++kx) {
                        int ix = ow + kx;
                        float xv = xsh[x_row + ix];
                        float wv = c_w1[w_base + ky * K1 + kx];
                        acc = fmaf(xv, wv, acc);
                    }
                }
                acc = relu(acc);
                m = acc > m ? acc : m;
            }
        }
        c1p_h[idx] = __float2half_rn(m);
    }
    __syncthreads();

    // ---- conv2 + relu + maxpool2 into float shared ----
    for (int idx = (int)threadIdx.x; idx < C2_OUT * H2P * W2P; idx += (int)blockDim.x) {
        int pw = idx % W2P;
        int t1 = idx / W2P;
        int ph = t1 % H2P;
        int oc = t1 / H2P;

        float m = 0.0f;
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            int oh = ph * 2 + dy;
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                int ow = pw * 2 + dx;

                float acc = c_b2[oc];
                int w_oc_base = oc * C2_IN * K2 * K2;

                #pragma unroll
                for (int ic = 0; ic < C2_IN; ++ic) {
                    int w_ic_base = w_oc_base + ic * K2 * K2;
                    int in_base = ic * H1P * W1P;
                    #pragma unroll
                    for (int ky = 0; ky < K2; ++ky) {
                        int iy = oh + ky;
                        int in_row = in_base + iy * W1P;
                        #pragma unroll
                        for (int kx = 0; kx < K2; ++kx) {
                            int ix = ow + kx;
                            float xv = __half2float(c1p_h[in_row + ix]);
                            float wv = c_w2[w_ic_base + ky * K2 + kx];
                            acc = fmaf(xv, wv, acc);
                        }
                    }
                }
                acc = relu(acc);
                m = acc > m ? acc : m;
            }
        }
        c2p[idx] = m;
    }
    __syncthreads();

    // ---- FC layers: warp-per-neuron, half2 weights, fp32 accumulate ----
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = (int)blockDim.x >> 5;

    // FC1: 400 -> 120 (400 divisible by 2)
    for (int i = warp; i < FC1_OUT; i += nwarps) {
        float sum = 0.0f;
        const half* wrow = w3h + i * FLAT;

        // use half2 over 400 => 200 half2
        const half2* w2p = reinterpret_cast<const half2*>(wrow);
        const float2* x2p = reinterpret_cast<const float2*>(c2p); // 400 floats => 200 float2 if aligned
        // Alignment of c2p is at least 4 bytes; float2 requires 8; shared allocations are aligned, so ok.
        int V = FLAT / 2; // 200
        for (int t = lane; t < V; t += 32) {
            half2 hw = LDG(w2p + t);
            float2 xf = x2p[t];
            float2 wf = __half22float2(hw);
            sum = fmaf(xf.x, wf.x, sum);
            sum = fmaf(xf.y, wf.y, sum);
        }

        sum = warp_sum(sum);
        if (lane == 0) fc1[i] = relu(sum + LDG(b3 + i));
    }
    __syncthreads();

    // FC2: 120 -> 84 (120 divisible by 2)
    for (int k = warp; k < FC2_OUT; k += nwarps) {
        float sum = 0.0f;
        const half* wrow = w4h + k * FC1_OUT;
        const half2* w2p = reinterpret_cast<const half2*>(wrow);
        const float2* x2p = reinterpret_cast<const float2*>(fc1); // 120 => 60 float2
        int V = FC1_OUT / 2; // 60
        for (int t = lane; t < V; t += 32) {
            half2 hw = LDG(w2p + t);
            float2 xf = x2p[t];
            float2 wf = __half22float2(hw);
            sum = fmaf(xf.x, wf.x, sum);
            sum = fmaf(xf.y, wf.y, sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) fc2[k] = relu(sum + LDG(b4 + k));
    }
    __syncthreads();

    // FC3: 84 -> C_out (84 divisible by 2)
    for (int j = warp; j < C_out; j += nwarps) {
        float sum = 0.0f;
        const half* wrow = w5h + j * FC2_OUT;
        const half2* w2p = reinterpret_cast<const half2*>(wrow);
        const float2* x2p = reinterpret_cast<const float2*>(fc2); // 84 => 42 float2
        int V = FC2_OUT / 2; // 42
        for (int t = lane; t < V; t += 32) {
            half2 hw = LDG(w2p + t);
            float2 xf = x2p[t];
            float2 wf = __half22float2(hw);
            sum = fmaf(xf.x, wf.x, sum);
            sum = fmaf(xf.y, wf.y, sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) y[n * C_out + j] = sum + LDG(b5 + j);
    }
}

// Persistent FP16 cache for FC weights (per-process, per-device).
struct FcCache {
    uintptr_t w3_ptr = 0;
    uintptr_t w4_ptr = 0;
    uintptr_t w5_ptr = 0;
    at::Tensor w3h;
    at::Tensor w4h;
    at::Tensor w5h;
};
static FcCache g_cache;

static inline at::Tensor ensure_f16_cached(torch::Tensor w, at::Tensor& wcache, uintptr_t& last_ptr) {
    uintptr_t p = (uintptr_t)w.data_ptr<float>();
    if (!wcache.defined() || p != last_ptr || wcache.numel() != w.numel() || wcache.device() != w.device()) {
        wcache = torch::empty(w.sizes(), torch::TensorOptions().device(w.device()).dtype(torch::kFloat16));
        int n = (int)w.numel();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        f32_to_f16_kernel<<<blocks, threads>>>(w.data_ptr<float>(), (half*)wcache.data_ptr<at::Half>(), n);
        last_ptr = p;
    }
    return wcache;
}

torch::Tensor lenet5_infer_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3,
    torch::Tensor w4, torch::Tensor b4,
    torch::Tensor w5, torch::Tensor b5
) {
    CHECK_CUDA(x);  CHECK_CONTIGUOUS(x);  CHECK_FLOAT(x);
    CHECK_CUDA(w1); CHECK_CONTIGUOUS(w1); CHECK_FLOAT(w1);
    CHECK_CUDA(b1); CHECK_CONTIGUOUS(b1); CHECK_FLOAT(b1);
    CHECK_CUDA(w2); CHECK_CONTIGUOUS(w2); CHECK_FLOAT(w2);
    CHECK_CUDA(b2); CHECK_CONTIGUOUS(b2); CHECK_FLOAT(b2);
    CHECK_CUDA(w3); CHECK_CONTIGUOUS(w3); CHECK_FLOAT(w3);
    CHECK_CUDA(b3); CHECK_CONTIGUOUS(b3); CHECK_FLOAT(b3);
    CHECK_CUDA(w4); CHECK_CONTIGUOUS(w4); CHECK_FLOAT(w4);
    CHECK_CUDA(b4); CHECK_CONTIGUOUS(b4); CHECK_FLOAT(b4);
    CHECK_CUDA(w5); CHECK_CONTIGUOUS(w5); CHECK_FLOAT(w5);
    CHECK_CUDA(b5); CHECK_CONTIGUOUS(b5); CHECK_FLOAT(b5);

    TORCH_CHECK(x.dim()==4 && x.size(1)==1 && x.size(2)==32 && x.size(3)==32, "x must be [N,1,32,32]");
    TORCH_CHECK(w1.dim()==4 && w1.size(0)==6 && w1.size(1)==1 && w1.size(2)==5 && w1.size(3)==5, "w1 must be [6,1,5,5]");
    TORCH_CHECK(b1.dim()==1 && b1.size(0)==6, "b1 must be [6]");
    TORCH_CHECK(w2.dim()==4 && w2.size(0)==16 && w2.size(1)==6 && w2.size(2)==5 && w2.size(3)==5, "w2 must be [16,6,5,5]");
    TORCH_CHECK(b2.dim()==1 && b2.size(0)==16, "b2 must be [16]");
    TORCH_CHECK(w3.dim()==2 && w3.size(0)==120 && w3.size(1)==400, "w3 must be [120,400]");
    TORCH_CHECK(b3.dim()==1 && b3.size(0)==120, "b3 must be [120]");
    TORCH_CHECK(w4.dim()==2 && w4.size(0)==84 && w4.size(1)==120, "w4 must be [84,120]");
    TORCH_CHECK(b4.dim()==1 && b4.size(0)==84, "b4 must be [84]");
    TORCH_CHECK(w5.dim()==2 && w5.size(1)==84, "w5 must be [Cout,84]");
    TORCH_CHECK(b5.dim()==1 && b5.size(0)==w5.size(0), "b5 must be [Cout]");

    maybe_copy_conv_to_const(w1, b1, w2, b2);

    // Ensure FP16 cached FC weights
    auto w3h = ensure_f16_cached(w3, g_cache.w3h, g_cache.w3_ptr);
    auto w4h = ensure_f16_cached(w4, g_cache.w4h, g_cache.w4_ptr);
    auto w5h = ensure_f16_cached(w5, g_cache.w5h, g_cache.w5_ptr);

    int64_t N = x.size(0);
    int64_t C_out = w5.size(0);
    auto y = torch::empty({N, C_out}, x.options());

    int threads = 128;
    dim3 block(threads);
    dim3 grid((unsigned)N);

    // shared:
    // xsh: 1024 floats
    // c1p_h: 1176 half
    // c2p: 400 floats
    // fc1: 120 floats
    // fc2: 84 floats
    size_t shmem = 0;
    shmem += (size_t)(H0*W0) * sizeof(float);
    shmem += (size_t)(C1_OUT*H1P*W1P) * sizeof(half);
    shmem += (size_t)(C2_OUT*H2P*W2P + FC1_OUT + FC2_OUT) * sizeof(float);

    lenet5_infer_kernel_opt5<<<grid, block, shmem>>>(
        x.data_ptr<float>(),
        (const half*)w3h.data_ptr<at::Half>(), b3.data_ptr<float>(),
        (const half*)w4h.data_ptr<at::Half>(), b4.data_ptr<float>(),
        (const half*)w5h.data_ptr<at::Half>(), b5.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)C_out
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor lenet5_infer_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3,
    torch::Tensor w4, torch::Tensor b4,
    torch::Tensor w5, torch::Tensor b5
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_lenet5_infer_opt5",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["lenet5_infer_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=int(num_classes))

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous()
            w1 = self.conv1.weight.contiguous()
            b1 = self.conv1.bias.contiguous()
            w2 = self.conv2.weight.contiguous()
            b2 = self.conv2.bias.contiguous()
            w3 = self.fc1.weight.contiguous()
            b3 = self.fc1.bias.contiguous()
            w4 = self.fc2.weight.contiguous()
            b4 = self.fc2.bias.contiguous()
            w5 = self.fc3.weight.contiguous()
            b5 = self.fc3.bias.contiguous()
            return custom_ops_lib.lenet5_infer_cuda(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x