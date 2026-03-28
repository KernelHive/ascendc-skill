import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Generic kernel: one block per batch element, one thread per hidden unit.
// Uses shared-memory persistent hidden buffers and pointer swap.
// One __syncthreads() per timestep.
template<int MAX_H>
__global__ __launch_bounds__(256, 2)
void lstm_final_c_generic_f32(
    const float* __restrict__ x_bti,   // [B, T, I]
    const float* __restrict__ h0_bh,   // [B, H]
    const float* __restrict__ c0_bh,   // [B, H]
    const float* __restrict__ w_ih,    // [4H, I]
    const float* __restrict__ w_hh,    // [4H, H]
    const float* __restrict__ b_ih,    // [4H] or nullptr
    const float* __restrict__ b_hh,    // [4H] or nullptr
    float* __restrict__ cT_bh,         // [B, H]
    int B, int T, int I, int H
) {
    int b = (int)blockIdx.x;
    int h = (int)threadIdx.x;
    if (b >= B || h >= H) return;

    extern __shared__ float shmem[]; // size: 2*H floats
    float* h_prev = shmem;
    float* h_next = shmem + H;

    // init
    h_prev[h] = h0_bh[(int64_t)b * H + h];
    float ct = c0_bh[(int64_t)b * H + h];
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const float* x_t = x_bti + ((int64_t)b * T + t) * (int64_t)I;

        int r_i = h;
        int r_f = H + h;
        int r_g = 2 * H + h;
        int r_o = 3 * H + h;

        const float* wih_i = w_ih + (int64_t)r_i * I;
        const float* wih_f = w_ih + (int64_t)r_f * I;
        const float* wih_g = w_ih + (int64_t)r_g * I;
        const float* wih_o = w_ih + (int64_t)r_o * I;

        float gi = 0.f, gf = 0.f, gg = 0.f, go = 0.f;

        // x contribution
        #pragma unroll 4
        for (int k = 0; k < 1024; ++k) { // bounded unroll-friendly loop; break at runtime
            if (k >= I) break;
            float xv = ldg_f32(x_t + k);
            gi = fmaf(xv, ldg_f32(wih_i + k), gi);
            gf = fmaf(xv, ldg_f32(wih_f + k), gf);
            gg = fmaf(xv, ldg_f32(wih_g + k), gg);
            go = fmaf(xv, ldg_f32(wih_o + k), go);
        }

        // recurrent contribution
        const float* whh_i = w_hh + (int64_t)r_i * H;
        const float* whh_f = w_hh + (int64_t)r_f * H;
        const float* whh_g = w_hh + (int64_t)r_g * H;
        const float* whh_o = w_hh + (int64_t)r_o * H;

        #pragma unroll 4
        for (int k = 0; k < MAX_H; ++k) {
            if (k >= H) break;
            float hv = h_prev[k];
            gi = fmaf(hv, ldg_f32(whh_i + k), gi);
            gf = fmaf(hv, ldg_f32(whh_f + k), gf);
            gg = fmaf(hv, ldg_f32(whh_g + k), gg);
            go = fmaf(hv, ldg_f32(whh_o + k), go);
        }

        if (b_ih) {
            gi += ldg_f32(b_ih + r_i); gf += ldg_f32(b_ih + r_f); gg += ldg_f32(b_ih + r_g); go += ldg_f32(b_ih + r_o);
        }
        if (b_hh) {
            gi += ldg_f32(b_hh + r_i); gf += ldg_f32(b_hh + r_f); gg += ldg_f32(b_hh + r_g); go += ldg_f32(b_hh + r_o);
        }

        float it = sigmoidf_fast(gi);
        float ft = sigmoidf_fast(gf);
        float gt = tanhf(gg);
        float ot = sigmoidf_fast(go);

        float ct_new = fmaf(ft, ct, it * gt);
        float ht_new = ot * tanhf(ct_new);

        h_next[h] = ht_new;
        ct = ct_new;

        __syncthreads(); // ensure h_next complete before swap
        float* tmp = h_prev;
        h_prev = h_next;
        h_next = tmp;
        // no second barrier needed since next iteration only reads h_prev after swap and barrier above ensures completion
    }

    cT_bh[(int64_t)b * H + h] = ct;
}

// Specialized kernel for I=128, H=256 (common in provided benchmark).
// Fully unrolled x loop (128) and H loop (256) to reduce control overhead.
// Still one barrier per timestep, shared pointer swap.
__global__ __launch_bounds__(256, 2)
void lstm_final_c_I128_H256_f32(
    const float* __restrict__ x_bti,   // [B, T, 128]
    const float* __restrict__ h0_bh,   // [B, 256]
    const float* __restrict__ c0_bh,   // [B, 256]
    const float* __restrict__ w_ih,    // [1024, 128]
    const float* __restrict__ w_hh,    // [1024, 256]
    const float* __restrict__ b_ih,    // [1024] or nullptr
    const float* __restrict__ b_hh,    // [1024] or nullptr
    float* __restrict__ cT_bh,         // [B, 256]
    int B, int T
) {
    int b = (int)blockIdx.x;
    int h = (int)threadIdx.x;
    if (b >= B || h >= 256) return;

    extern __shared__ float shmem[]; // 2*256 floats
    float* h_prev = shmem;
    float* h_next = shmem + 256;

    h_prev[h] = h0_bh[(int64_t)b * 256 + h];
    float ct = c0_bh[(int64_t)b * 256 + h];
    __syncthreads();

    #pragma unroll 1
    for (int t = 0; t < T; ++t) {
        const float* x_t = x_bti + ((int64_t)b * T + t) * 128;

        int r_i = h;
        int r_f = 256 + h;
        int r_g = 512 + h;
        int r_o = 768 + h;

        const float* wih_i = w_ih + (int64_t)r_i * 128;
        const float* wih_f = w_ih + (int64_t)r_f * 128;
        const float* wih_g = w_ih + (int64_t)r_g * 128;
        const float* wih_o = w_ih + (int64_t)r_o * 128;

        float gi = 0.f, gf = 0.f, gg = 0.f, go = 0.f;

        #pragma unroll 8
        for (int k = 0; k < 128; ++k) {
            float xv = ldg_f32(x_t + k);
            gi = fmaf(xv, ldg_f32(wih_i + k), gi);
            gf = fmaf(xv, ldg_f32(wih_f + k), gf);
            gg = fmaf(xv, ldg_f32(wih_g + k), gg);
            go = fmaf(xv, ldg_f32(wih_o + k), go);
        }

        const float* whh_i = w_hh + (int64_t)r_i * 256;
        const float* whh_f = w_hh + (int64_t)r_f * 256;
        const float* whh_g = w_hh + (int64_t)r_g * 256;
        const float* whh_o = w_hh + (int64_t)r_o * 256;

        #pragma unroll 8
        for (int k = 0; k < 256; ++k) {
            float hv = h_prev[k];
            gi = fmaf(hv, ldg_f32(whh_i + k), gi);
            gf = fmaf(hv, ldg_f32(whh_f + k), gf);
            gg = fmaf(hv, ldg_f32(whh_g + k), gg);
            go = fmaf(hv, ldg_f32(whh_o + k), go);
        }

        if (b_ih) {
            gi += ldg_f32(b_ih + r_i); gf += ldg_f32(b_ih + r_f); gg += ldg_f32(b_ih + r_g); go += ldg_f32(b_ih + r_o);
        }
        if (b_hh) {
            gi += ldg_f32(b_hh + r_i); gf += ldg_f32(b_hh + r_f); gg += ldg_f32(b_hh + r_g); go += ldg_f32(b_hh + r_o);
        }

        float it = sigmoidf_fast(gi);
        float ft = sigmoidf_fast(gf);
        float gt = tanhf(gg);
        float ot = sigmoidf_fast(go);

        float ct_new = fmaf(ft, ct, it * gt);
        float ht_new = ot * tanhf(ct_new);

        h_next[h] = ht_new;
        ct = ct_new;

        __syncthreads();
        float* tmp = h_prev;
        h_prev = h_next;
        h_next = tmp;
    }

    cT_bh[(int64_t)b * 256 + h] = ct;
}

torch::Tensor lstm_cn_f32_cuda(
    torch::Tensor x,           // [B, T, I0]
    torch::Tensor h0,          // [L, B, H]
    torch::Tensor c0,          // [L, B, H]
    std::vector<torch::Tensor> w_ih, // L tensors: [4H, I_l]
    std::vector<torch::Tensor> w_hh, // L tensors: [4H, H]
    std::vector<torch::Tensor> b_ih, // L tensors or empty: [4H]
    std::vector<torch::Tensor> b_hh  // L tensors or empty: [4H]
) {
    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(c0);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(h0.scalar_type() == at::kFloat, "h0 must be float32");
    TORCH_CHECK(c0.scalar_type() == at::kFloat, "c0 must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be [B, T, I0]");
    TORCH_CHECK(h0.dim() == 3 && c0.dim() == 3, "h0/c0 must be [L, B, H]");
    TORCH_CHECK(h0.sizes() == c0.sizes(), "h0 and c0 must have same shape");

    const int64_t L = h0.size(0);
    const int64_t B = h0.size(1);
    const int64_t H = h0.size(2);
    const int64_t T = x.size(1);
    const int64_t I0 = x.size(2);
    TORCH_CHECK(x.size(0) == B, "batch mismatch");
    TORCH_CHECK(T >= 1, "T must be >= 1");
    TORCH_CHECK(L == 1, "custom lstm_cn_f32_cuda supports num_layers==1 only for full correctness");
    TORCH_CHECK(H <= 256, "hidden_size H must be <= 256 for this kernel");

    TORCH_CHECK((int64_t)w_ih.size() == L, "w_ih must have L tensors");
    TORCH_CHECK((int64_t)w_hh.size() == L, "w_hh must have L tensors");
    TORCH_CHECK(b_ih.empty() || (int64_t)b_ih.size() == L, "b_ih must be empty or L tensors");
    TORCH_CHECK(b_hh.empty() || (int64_t)b_hh.size() == L, "b_hh must be empty or L tensors");

    auto cn = torch::empty_like(c0);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    auto wih = w_ih[0];
    auto whh = w_hh[0];
    CHECK_INPUT(wih);
    CHECK_INPUT(whh);
    TORCH_CHECK(wih.scalar_type() == at::kFloat && whh.scalar_type() == at::kFloat, "weights must be float32");
    TORCH_CHECK(wih.dim() == 2 && whh.dim() == 2, "weights must be 2D");
    TORCH_CHECK(wih.size(0) == 4 * H, "w_ih first dim must be 4H");
    TORCH_CHECK(whh.size(0) == 4 * H && whh.size(1) == H, "w_hh must be [4H, H]");
    TORCH_CHECK(wih.size(1) == I0, "w_ih second dim mismatch");

    const float* b_ih_ptr = nullptr;
    const float* b_hh_ptr = nullptr;
    if (!b_ih.empty()) {
        auto bih = b_ih[0];
        CHECK_INPUT(bih);
        TORCH_CHECK(bih.scalar_type() == at::kFloat && bih.dim() == 1 && bih.size(0) == 4 * H, "b_ih must be [4H]");
        b_ih_ptr = (const float*)bih.data_ptr<float>();
    }
    if (!b_hh.empty()) {
        auto bhh = b_hh[0];
        CHECK_INPUT(bhh);
        TORCH_CHECK(bhh.scalar_type() == at::kFloat && bhh.dim() == 1 && bhh.size(0) == 4 * H, "b_hh must be [4H]");
        b_hh_ptr = (const float*)bhh.data_ptr<float>();
    }

    auto h0_l = h0.select(0, 0).contiguous(); // [B, H]
    auto c0_l = c0.select(0, 0).contiguous(); // [B, H]
    auto x_c = x.contiguous();

    auto cT = torch::empty({B, H}, x.options());

    dim3 blocks((unsigned)B);
    const int threads = 256;
    size_t shmem_bytes = (size_t)(2 * H) * sizeof(float);

    // Fast path specialization
    if (I0 == 128 && H == 256) {
        size_t shmem_256 = (size_t)(2 * 256) * sizeof(float);
        lstm_final_c_I128_H256_f32<<<blocks, threads, shmem_256, stream>>>(
            (const float*)x_c.data_ptr<float>(),
            (const float*)h0_l.data_ptr<float>(),
            (const float*)c0_l.data_ptr<float>(),
            (const float*)wih.data_ptr<float>(),
            (const float*)whh.data_ptr<float>(),
            b_ih_ptr,
            b_hh_ptr,
            (float*)cT.data_ptr<float>(),
            (int)B, (int)T
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // Generic path with MAX_H=256
        lstm_final_c_generic_f32<256><<<blocks, threads, shmem_bytes, stream>>>(
            (const float*)x_c.data_ptr<float>(),
            (const float*)h0_l.data_ptr<float>(),
            (const float*)c0_l.data_ptr<float>(),
            (const float*)wih.data_ptr<float>(),
            (const float*)whh.data_ptr<float>(),
            b_ih_ptr,
            b_hh_ptr,
            (float*)cT.data_ptr<float>(),
            (int)B, (int)T, (int)I0, (int)H
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    cn.select(0, 0).copy_(cT);
    return cn;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor lstm_cn_f32_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh
);
"""

_ext_name = "custom_ops_lib_lstm_cn_v4"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["lstm_cn_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    Uses a custom CUDA op to produce LSTM cn (cell state) when safe.
    Falls back to nn.LSTM for full generality.
    Note: custom path supports num_layers==1 only (exact semantics).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        try_custom = (
            x.is_cuda and h0.is_cuda and c0.is_cuda
            and x.dtype == torch.float32 and h0.dtype == torch.float32 and c0.dtype == torch.float32
            and x.is_contiguous() and h0.is_contiguous() and c0.is_contiguous()
            and x.dim() == 3 and h0.dim() == 3 and c0.dim() == 3
            and self.lstm.batch_first is True
            and self.lstm.bidirectional is False
            and float(self.lstm.dropout) == 0.0
            and int(self.lstm.num_layers) == 1
            and h0.size(0) == c0.size(0) == 1
            and h0.size(1) == c0.size(1) == x.size(0)
            and h0.size(2) == c0.size(2) == self.lstm.hidden_size
            and self.lstm.hidden_size <= 256
        )

        if try_custom:
            w_ih = [getattr(self.lstm, "weight_ih_l0")]
            w_hh = [getattr(self.lstm, "weight_hh_l0")]
            b_ih_t = getattr(self.lstm, "bias_ih_l0", None)
            b_hh_t = getattr(self.lstm, "bias_hh_l0", None)
            b_ih = [b_ih_t] if b_ih_t is not None else []
            b_hh = [b_hh_t] if b_hh_t is not None else []

            if (
                w_ih[0].is_cuda and w_hh[0].is_cuda
                and w_ih[0].dtype == torch.float32 and w_hh[0].dtype == torch.float32
                and w_ih[0].is_contiguous() and w_hh[0].is_contiguous()
                and (not b_ih or (b_ih[0].is_cuda and b_ih[0].dtype == torch.float32 and b_ih[0].is_contiguous()))
                and (not b_hh or (b_hh[0].is_cuda and b_hh[0].dtype == torch.float32 and b_hh[0].is_contiguous()))
            ):
                return self.custom_ops_lib.lstm_cn_f32_cuda(x, h0, c0, w_ih, w_hh, b_ih, b_hh)

        out, state = self.lstm(x, (h0, c0))
        _ = self.fc(out[:, -1, :])
        return state[1]