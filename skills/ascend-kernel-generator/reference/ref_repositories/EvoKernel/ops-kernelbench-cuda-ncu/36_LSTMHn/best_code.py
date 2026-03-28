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

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

/*
  Optimized single-layer full-sequence LSTM forward producing final h_T.
  Design goals vs baseline:
    - Remove global staging buffer; keep h_prev in shared memory for the whole block (one batch element).
    - Reduce barriers: only one __syncthreads() per timestep (publish next h_prev).
    - Reduce register pressure via smaller CTA (128) and launch bounds.
    - Use read-only cache for x and weights via __ldg.
  Mapping: one block per batch, threads cover hidden rows (tid < H). For H=256 and CTA=128, each thread computes 2 rows.
*/
template<int THREADS>
__global__ __launch_bounds__(THREADS, 3)
void lstm_layer_final_h_f32_shprev_kernel(
    const float* __restrict__ x_bti,   // [B, T, I]
    const float* __restrict__ h0_bh,   // [B, H]
    const float* __restrict__ c0_bh,   // [B, H]
    const float* __restrict__ w_ih,    // [4H, I]
    const float* __restrict__ w_hh,    // [4H, H]
    const float* __restrict__ b_ih,    // [4H] or nullptr
    const float* __restrict__ b_hh,    // [4H] or nullptr
    float* __restrict__ hT_bh,         // [B, H]
    int B, int T, int I, int H
) {
    int b = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (b >= B) return;

    extern __shared__ float shmem[]; // size H floats
    float* sh_h = shmem;

    // Load h0 into sh_h
    for (int k = tid; k < H; k += THREADS) {
        sh_h[k] = ldg_f32(h0_bh + (int64_t)b * H + k);
    }
    __syncthreads();

    // Each thread may own multiple rows (row = tid + m*THREADS)
    float ct0 = 0.f, ct1 = 0.f;
    int row0 = tid;
    int row1 = tid + THREADS;
    bool do0 = (row0 < H);
    bool do1 = (row1 < H);

    if (do0) ct0 = ldg_f32(c0_bh + (int64_t)b * H + row0);
    if (do1) ct1 = ldg_f32(c0_bh + (int64_t)b * H + row1);

    for (int t = 0; t < T; ++t) {
        const float* x_t = x_bti + ((int64_t)b * T + (int64_t)t) * (int64_t)I;

        // Compute for row0
        float gi0=0.f, gf0=0.f, gg0=0.f, go0=0.f;
        // Compute for row1
        float gi1=0.f, gf1=0.f, gg1=0.f, go1=0.f;

        if (do0 || do1) {
            // Input term W_ih * x_t (I=128 here; keep simple loop, rely on L1/readonly cache)
            // Recurrent term W_hh * sh_h
            // We fuse gate accumulations; minimize temporaries.
            if (do0) {
                int r_i = row0;
                int r_f = H + row0;
                int r_g = 2 * H + row0;
                int r_o = 3 * H + row0;

                const float* wih_i = w_ih + (int64_t)r_i * I;
                const float* wih_f = w_ih + (int64_t)r_f * I;
                const float* wih_g = w_ih + (int64_t)r_g * I;
                const float* wih_o = w_ih + (int64_t)r_o * I;

                #pragma unroll 4
                for (int k = 0; k < 128; ++k) { // specialization-friendly; wrapper enforces I==128 for fast path
                    float xv = ldg_f32(x_t + k);
                    gi0 = fmaf(xv, ldg_f32(wih_i + k), gi0);
                    gf0 = fmaf(xv, ldg_f32(wih_f + k), gf0);
                    gg0 = fmaf(xv, ldg_f32(wih_g + k), gg0);
                    go0 = fmaf(xv, ldg_f32(wih_o + k), go0);
                }

                const float* whh_i = w_hh + (int64_t)r_i * H;
                const float* whh_f = w_hh + (int64_t)r_f * H;
                const float* whh_g = w_hh + (int64_t)r_g * H;
                const float* whh_o = w_hh + (int64_t)r_o * H;

                #pragma unroll 4
                for (int k = 0; k < 256; ++k) { // specialization-friendly; wrapper enforces H==256 for fast path
                    float hv = sh_h[k];
                    gi0 = fmaf(hv, ldg_f32(whh_i + k), gi0);
                    gf0 = fmaf(hv, ldg_f32(whh_f + k), gf0);
                    gg0 = fmaf(hv, ldg_f32(whh_g + k), gg0);
                    go0 = fmaf(hv, ldg_f32(whh_o + k), go0);
                }

                if (b_ih) { gi0 += ldg_f32(b_ih + r_i); gf0 += ldg_f32(b_ih + r_f); gg0 += ldg_f32(b_ih + r_g); go0 += ldg_f32(b_ih + r_o); }
                if (b_hh) { gi0 += ldg_f32(b_hh + r_i); gf0 += ldg_f32(b_hh + r_f); gg0 += ldg_f32(b_hh + r_g); go0 += ldg_f32(b_hh + r_o); }

                float it = sigmoidf_fast(gi0);
                float ft = sigmoidf_fast(gf0);
                float gt = tanhf(gg0);
                float ot = sigmoidf_fast(go0);

                float ct_new = ft * ct0 + it * gt;
                float ht_new = ot * tanhf(ct_new);
                ct0 = ct_new;

                sh_h[row0] = ht_new;
            }

            if (do1) {
                int r_i = row1;
                int r_f = H + row1;
                int r_g = 2 * H + row1;
                int r_o = 3 * H + row1;

                const float* wih_i = w_ih + (int64_t)r_i * I;
                const float* wih_f = w_ih + (int64_t)r_f * I;
                const float* wih_g = w_ih + (int64_t)r_g * I;
                const float* wih_o = w_ih + (int64_t)r_o * I;

                #pragma unroll 4
                for (int k = 0; k < 128; ++k) {
                    float xv = ldg_f32(x_t + k);
                    gi1 = fmaf(xv, ldg_f32(wih_i + k), gi1);
                    gf1 = fmaf(xv, ldg_f32(wih_f + k), gf1);
                    gg1 = fmaf(xv, ldg_f32(wih_g + k), gg1);
                    go1 = fmaf(xv, ldg_f32(wih_o + k), go1);
                }

                const float* whh_i = w_hh + (int64_t)r_i * H;
                const float* whh_f = w_hh + (int64_t)r_f * H;
                const float* whh_g = w_hh + (int64_t)r_g * H;
                const float* whh_o = w_hh + (int64_t)r_o * H;

                #pragma unroll 4
                for (int k = 0; k < 256; ++k) {
                    float hv = sh_h[k];
                    gi1 = fmaf(hv, ldg_f32(whh_i + k), gi1);
                    gf1 = fmaf(hv, ldg_f32(whh_f + k), gf1);
                    gg1 = fmaf(hv, ldg_f32(whh_g + k), gg1);
                    go1 = fmaf(hv, ldg_f32(whh_o + k), go1);
                }

                if (b_ih) { gi1 += ldg_f32(b_ih + r_i); gf1 += ldg_f32(b_ih + r_f); gg1 += ldg_f32(b_ih + r_g); go1 += ldg_f32(b_ih + r_o); }
                if (b_hh) { gi1 += ldg_f32(b_hh + r_i); gf1 += ldg_f32(b_hh + r_f); gg1 += ldg_f32(b_hh + r_g); go1 += ldg_f32(b_hh + r_o); }

                float it = sigmoidf_fast(gi1);
                float ft = sigmoidf_fast(gf1);
                float gt = tanhf(gg1);
                float ot = sigmoidf_fast(go1);

                float ct_new = ft * ct1 + it * gt;
                float ht_new = ot * tanhf(ct_new);
                ct1 = ct_new;

                sh_h[row1] = ht_new;
            }
        }

        // Publish sh_h for next timestep
        __syncthreads();
    }

    // Store final h
    for (int k = tid; k < H; k += THREADS) {
        hT_bh[(int64_t)b * H + k] = sh_h[k];
    }
}

torch::Tensor lstm_hn_f32_cuda(
    torch::Tensor x,                 // [B, T, I0]
    torch::Tensor h0,                // [L, B, H]
    torch::Tensor c0,                // [L, B, H]
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
    TORCH_CHECK((int64_t)w_ih.size() == L, "w_ih must have L tensors");
    TORCH_CHECK((int64_t)w_hh.size() == L, "w_hh must have L tensors");
    TORCH_CHECK(b_ih.empty() || (int64_t)b_ih.size() == L, "b_ih must be empty or L tensors");
    TORCH_CHECK(b_hh.empty() || (int64_t)b_hh.size() == L, "b_hh must be empty or L tensors");

    // Output: hn [L, B, H]
    auto hn = torch::empty_like(h0);
    torch::Tensor hT = torch::empty({B, H}, x.options());

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    TORCH_CHECK(L == 1, "custom lstm_hn_f32_cuda supports num_layers==1 only");
    // Fast specialization for the provided workload shape
    TORCH_CHECK(H == 256, "fast kernel requires hidden_size==256");
    TORCH_CHECK(I0 == 128, "fast kernel requires input_size==128");

    for (int64_t l = 0; l < L; ++l) {
        auto wih = w_ih[l];
        auto whh = w_hh[l];
        CHECK_INPUT(wih);
        CHECK_INPUT(whh);
        TORCH_CHECK(wih.scalar_type() == at::kFloat && whh.scalar_type() == at::kFloat, "weights must be float32");
        TORCH_CHECK(wih.dim() == 2 && whh.dim() == 2, "weights must be 2D");
        TORCH_CHECK(wih.size(0) == 4 * H, "w_ih[l] first dim must be 4H");
        TORCH_CHECK(whh.size(0) == 4 * H && whh.size(1) == H, "w_hh[l] must be [4H, H]");
        TORCH_CHECK(wih.size(1) == I0, "w_ih[l] second dim mismatch");

        const float* b_ih_ptr = nullptr;
        const float* b_hh_ptr = nullptr;
        if (!b_ih.empty()) {
            auto bih = b_ih[l];
            CHECK_INPUT(bih);
            TORCH_CHECK(bih.scalar_type() == at::kFloat && bih.dim() == 1 && bih.size(0) == 4 * H, "b_ih[l] must be [4H]");
            b_ih_ptr = (const float*)bih.data_ptr<float>();
        }
        if (!b_hh.empty()) {
            auto bhh = b_hh[l];
            CHECK_INPUT(bhh);
            TORCH_CHECK(bhh.scalar_type() == at::kFloat && bhh.dim() == 1 && bhh.size(0) == 4 * H, "b_hh[l] must be [4H]");
            b_hh_ptr = (const float*)bhh.data_ptr<float>();
        }

        auto h0_l = h0.select(0, l).contiguous(); // [B, H]
        auto c0_l = c0.select(0, l).contiguous(); // [B, H]
        auto x_c = x.contiguous();

        constexpr int THREADS = 128;
        dim3 blocks((unsigned)B);
        size_t shmem = (size_t)H * sizeof(float); // sh_h only

        lstm_layer_final_h_f32_shprev_kernel<THREADS><<<blocks, THREADS, shmem, stream>>>(
            (const float*)x_c.data_ptr<float>(),
            (const float*)h0_l.data_ptr<float>(),
            (const float*)c0_l.data_ptr<float>(),
            (const float*)wih.data_ptr<float>(),
            (const float*)whh.data_ptr<float>(),
            b_ih_ptr,
            b_hh_ptr,
            (float*)hT.data_ptr<float>(),
            (int)B, (int)T, (int)I0, (int)H
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        hn.select(0, l).copy_(hT);
    }

    return hn;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor lstm_hn_f32_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh
);
"""

_ext_name = "custom_ops_lib_lstm_hn_v3"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["lstm_hn_f32_cuda"],
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
    Custom CUDA path returns h_n only (matches model's returned value).
    Fast path specializes the workload in this prompt:
      - float32 CUDA contiguous
      - batch_first=True, bidirectional=False, dropout==0
      - num_layers == 1
      - input_size == 128
      - hidden_size == 256
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
            and int(self.lstm.hidden_size) == 256
            and int(self.lstm.input_size) == 128
            and int(x.size(2)) == 128
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
                hn = self.custom_ops_lib.lstm_hn_f32_cuda(x, h0, c0, w_ih, w_hh, b_ih, b_hh)
                _ = self.fc(hn[0])
                return hn

        out, state = self.lstm(x, (h0, c0))
        _ = self.fc(out[:, -1, :])
        return state[0]