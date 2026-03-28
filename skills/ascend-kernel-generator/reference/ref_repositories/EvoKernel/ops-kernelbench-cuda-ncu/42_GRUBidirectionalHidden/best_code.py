import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

static __device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
static __device__ __forceinline__ float tanhf_fast(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float warp_sum(float v) {
    // full mask
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Generic transpose for fp16
__global__ void transpose_f16_kernel(const half* __restrict__ in, half* __restrict__ out, int M, int N) {
    int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int m = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (m < M && n < N) out[n * M + m] = in[m * N + n];
}

// Prepack weights to gate-major contiguous K:
// w_ih: (2,3H,I)  -> wihG: (2,3,H,I) where last dim is K=I contiguous
// w_hh: (2,3H,H)  -> whhG: (2,3,H,H) where last dim is K=H contiguous
__global__ void prepack_wih_gate_major(const half* __restrict__ wih, half* __restrict__ out, int I, int H) {
    // index over dir, gate, h, k
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long long total = (long long)2 * 3 * H * I;
    if ((long long)idx >= total) return;
    int k = idx % I;
    int tmp = idx / I;
    int h = tmp % H;
    tmp /= H;
    int g = tmp % 3;
    int dir = tmp / 3;

    // wih layout: (dir, 3H, I) row-major => ((dir*3H + g*H + h)*I + k)
    out[idx] = __ldg(&wih[((dir * (3 * H) + g * H + h) * I) + k]);
}

__global__ void prepack_whh_gate_major(const half* __restrict__ whh, half* __restrict__ out, int H) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long long total = (long long)2 * 3 * H * H;
    if ((long long)idx >= total) return;
    int k = idx % H;
    int tmp = idx / H;
    int h = tmp % H;
    tmp /= H;
    int g = tmp % 3;
    int dir = tmp / 3;

    // whh layout: (dir, 3H, H) => ((dir*3H + g*H + h)*H + k)
    out[idx] = __ldg(&whh[((dir * (3 * H) + g * H + h) * H) + k]);
}

// Fused bidirectional GRU final hidden, single-layer, fp16.
// Uses warp-per-hidden-unit decomposition: each warp computes one hidden index h.
// Block is fixed at 256 threads => 8 warps => computes 8 h's at a time; loops over h tiles.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2) void gru_bi_hidden_warp_f16_kernel(
    const half* __restrict__ x,        // (T,B,I)
    const half* __restrict__ h0,       // (2,B,H)
    const half* __restrict__ wihG,     // (2,3,H,I) flattened
    const half* __restrict__ whhG,     // (2,3,H,H) flattened
    const half* __restrict__ bsum,     // (2,3,H) flattened (b_ih + b_hh)
    half* __restrict__ hn,             // (2,B,H)
    int T, int B, int I, int H
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    extern __shared__ half shmem[];
    half* sh_x  = shmem;          // I
    half* sh_hf = sh_x + I;       // H
    half* sh_hb = sh_hf + H;      // H

    // init hidden
    for (int h = threadIdx.x; h < H; h += THREADS) {
        sh_hf[h] = h0[(0 * B + b) * H + h];
        sh_hb[h] = h0[(1 * B + b) * H + h];
    }
    __syncthreads();

    int warp = threadIdx.x >> 5;   // 0..7
    int lane = threadIdx.x & 31;   // 0..31
    int warps_per_block = THREADS / 32; // 8

    // helper lambda: compute one direction step given x_t in shared and hidden in shared
    auto step_dir = [&](int dir, half* sh_h) {
        // loop over hidden indices assigned to warps
        for (int h_base = 0; h_base < H; h_base += warps_per_block) {
            int h = h_base + warp;
            if (h >= H) continue;

            // base offsets for weights/bias
            // wihG index: (((dir*3 + g)*H + h)*I + k)
            // whhG index: (((dir*3 + g)*H + h)*H + k)
            // bsum index: ((dir*3 + g)*H + h)
            int base_b = (dir * 3) * H + h;

            // gate accumulators
            float ir = (lane == 0) ? __half2float(__ldg(&bsum[base_b + 0 * H])) : 0.0f;
            float iz = (lane == 0) ? __half2float(__ldg(&bsum[base_b + 1 * H])) : 0.0f;
            float in_ = (lane == 0) ? __half2float(__ldg(&bsum[base_b + 2 * H])) : 0.0f;

            // x contribution (stride over I)
            int base_wx = (dir * 3 * H + h) * I; // for gate 0, then + g*H*I
            for (int k = lane; k < I; k += 32) {
                float xv = __half2float(sh_x[k]);
                ir += xv * __half2float(__ldg(&wihG[(0 * H) * I + base_wx + k]));                 // g=0
                iz += xv * __half2float(__ldg(&wihG[(1 * H) * I + base_wx + k]));                 // g=1
                in_ += xv * __half2float(__ldg(&wihG[(2 * H) * I + base_wx + k]));                // g=2
            }
            ir = warp_sum(ir);
            iz = warp_sum(iz);
            in_ = warp_sum(in_);

            // h contribution (stride over H)
            float hr = 0.0f, hz = 0.0f, hn_ = 0.0f;
            int base_wh = (dir * 3 * H + h) * H; // for gate 0, then + g*H*H
            for (int k = lane; k < H; k += 32) {
                float hv = __half2float(sh_h[k]);
                hr += hv * __half2float(__ldg(&whhG[(0 * H) * H + base_wh + k]));
                hz += hv * __half2float(__ldg(&whhG[(1 * H) * H + base_wh + k]));
                hn_ += hv * __half2float(__ldg(&whhG[(2 * H) * H + base_wh + k]));
            }
            hr = warp_sum(hr);
            hz = warp_sum(hz);
            hn_ = warp_sum(hn_);

            if (lane == 0) {
                float r = sigmoidf_fast(ir + hr);
                float z = sigmoidf_fast(iz + hz);
                float nval = tanhf_fast(in_ + r * hn_);
                float hprev = __half2float(sh_h[h]);
                float hnew = (1.0f - z) * nval + z * hprev;
                sh_h[h] = __float2half_rn(hnew);
            }
        }
    };

    for (int t = 0; t < T; ++t) {
        // forward: load x[t]
        {
            const half* x_ptr = x + (t * B + b) * I;
            for (int i = threadIdx.x; i < I; i += THREADS) sh_x[i] = x_ptr[i];
            __syncthreads();
            step_dir(0, sh_hf);
            __syncthreads();
        }

        // backward: load x[T-1-t]
        {
            int tt = (T - 1 - t);
            const half* x_ptr = x + (tt * B + b) * I;
            for (int i = threadIdx.x; i < I; i += THREADS) sh_x[i] = x_ptr[i];
            __syncthreads();
            step_dir(1, sh_hb);
            __syncthreads();
        }
    }

    // store final
    for (int h = threadIdx.x; h < H; h += THREADS) {
        hn[(0 * B + b) * H + h] = sh_hf[h];
        hn[(1 * B + b) * H + h] = sh_hb[h];
    }
}

torch::Tensor gru_bi_hidden_single_layer_f16_warp_cuda(
    torch::Tensor x,      // (T,B,I) half
    torch::Tensor h0,     // (2,B,H) half
    torch::Tensor wihG,   // (2,3,H,I) half (flattened)
    torch::Tensor whhG,   // (2,3,H,H) half (flattened)
    torch::Tensor bsum    // (2,3,H) half (flattened)
) {
    TORCH_CHECK(x.is_cuda() && h0.is_cuda(), "x/h0 must be CUDA");
    TORCH_CHECK(wihG.is_cuda() && whhG.is_cuda() && bsum.is_cuda(), "params must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16 && h0.dtype() == torch::kFloat16, "x/h0 must be float16");
    TORCH_CHECK(wihG.dtype() == torch::kFloat16 && whhG.dtype() == torch::kFloat16 && bsum.dtype() == torch::kFloat16, "params must be float16");
    TORCH_CHECK(x.dim() == 3, "x must be (T,B,I)");
    TORCH_CHECK(h0.dim() == 3 && h0.size(0) == 2, "h0 must be (2,B,H)");

    auto xc = x.contiguous();
    auto h0c = h0.contiguous();
    auto wihc = wihG.contiguous();
    auto whhc = whhG.contiguous();
    auto bsc = bsum.contiguous();

    int T = (int)xc.size(0);
    int B = (int)xc.size(1);
    int I = (int)xc.size(2);
    TORCH_CHECK((int)h0c.size(1) == B, "h0 batch mismatch");
    int H = (int)h0c.size(2);

    TORCH_CHECK(wihc.numel() == (long long)2 * 3 * H * I, "wihG numel mismatch");
    TORCH_CHECK(whhc.numel() == (long long)2 * 3 * H * H, "whhG numel mismatch");
    TORCH_CHECK(bsc.numel() == (long long)2 * 3 * H, "bsum numel mismatch");

    auto hn = torch::empty({2, B, H}, xc.options());

    constexpr int THREADS = 256;
    dim3 grid(B);
    dim3 block(THREADS);
    size_t shmem = (size_t)(I + 2 * H) * sizeof(half);
    TORCH_CHECK(shmem <= 96 * 1024, "shared memory too large for (I + 2H)");

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    gru_bi_hidden_warp_f16_kernel<THREADS><<<grid, block, shmem, stream>>>(
        (const half*)xc.data_ptr<at::Half>(),
        (const half*)h0c.data_ptr<at::Half>(),
        (const half*)wihc.data_ptr<at::Half>(),
        (const half*)whhc.data_ptr<at::Half>(),
        (const half*)bsc.data_ptr<at::Half>(),
        (half*)hn.data_ptr<at::Half>(),
        T, B, I, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return hn;
}

std::vector<torch::Tensor> gru_bi_prepack_gate_major_f16_cuda(
    torch::Tensor w_ih, // (2,3H,I)
    torch::Tensor w_hh  // (2,3H,H)
) {
    TORCH_CHECK(w_ih.is_cuda() && w_hh.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(w_ih.dtype() == torch::kFloat16 && w_hh.dtype() == torch::kFloat16, "weights must be float16");
    TORCH_CHECK(w_ih.dim() == 3 && w_hh.dim() == 3, "weights must be 3D");
    TORCH_CHECK((int)w_ih.size(0) == 2 && (int)w_hh.size(0) == 2, "first dim must be 2");
    TORCH_CHECK((int)w_ih.size(1) % 3 == 0, "w_ih second dim must be 3H");

    int threeH = (int)w_ih.size(1);
    int I = (int)w_ih.size(2);
    int H = threeH / 3;
    TORCH_CHECK((int)w_hh.size(1) == threeH && (int)w_hh.size(2) == H, "w_hh must be (2,3H,H)");

    auto wihc = w_ih.contiguous();
    auto whhc = w_hh.contiguous();

    auto wihG = torch::empty({2, 3, H, I}, wihc.options());
    auto whhG = torch::empty({2, 3, H, H}, whhc.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int threads = 256;

    {
        long long total = (long long)2 * 3 * H * I;
        int blocks = (int)((total + threads - 1) / threads);
        prepack_wih_gate_major<<<blocks, threads, 0, stream>>>(
            (const half*)wihc.data_ptr<at::Half>(),
            (half*)wihG.data_ptr<at::Half>(),
            I, H
        );
    }
    {
        long long total = (long long)2 * 3 * H * H;
        int blocks = (int)((total + threads - 1) / threads);
        prepack_whh_gate_major<<<blocks, threads, 0, stream>>>(
            (const half*)whhc.data_ptr<at::Half>(),
            (half*)whhG.data_ptr<at::Half>(),
            H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {wihG, whhG};
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gru_bi_hidden_single_layer_f16_warp_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor wihG,
    torch::Tensor whhG,
    torch::Tensor bsum
);
std::vector<torch::Tensor> gru_bi_prepack_gate_major_f16_cuda(
    torch::Tensor w_ih,
    torch::Tensor w_hh
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gru_bi_hidden_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "gru_bi_hidden_single_layer_f16_warp_cuda",
        "gru_bi_prepack_gate_major_f16_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Operator replacement for `gru_bidirectional_hidden` (returns h_n only).

    Custom CUDA path supports ONLY:
      - CUDA float16
      - batch_first=False
      - bias=True
      - bidirectional=True
      - dropout=0
      - num_layers=1
      - returns h_n with shape (2,B,H)

    Otherwise falls back to nn.GRU.
    """
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=0.0,
            bidirectional=True,
        ).half()

        self.custom_ops_lib = custom_ops_lib

        self._packed_wihG = None
        self._packed_whhG = None
        self._wih_ptrs = None
        self._whh_ptrs = None
        self._packed_shapes = None  # (I,H,device)

    def _maybe_repack(self):
        H = self.hidden_size
        I = self.input_size
        dev = self.gru.weight_ih_l0.device

        w_ih = torch.stack([self.gru.weight_ih_l0, self.gru.weight_ih_l0_reverse], dim=0).contiguous()
        w_hh = torch.stack([self.gru.weight_hh_l0, self.gru.weight_hh_l0_reverse], dim=0).contiguous()

        wih_ptrs = (w_ih.storage().data_ptr(), int(w_ih.storage_offset()))
        whh_ptrs = (w_hh.storage().data_ptr(), int(w_hh.storage_offset()))
        key_shapes = (I, H, dev.index if dev.type == "cuda" else -1)

        if (
            self._packed_wihG is None or
            self._packed_whhG is None or
            self._wih_ptrs != wih_ptrs or
            self._whh_ptrs != whh_ptrs or
            self._packed_shapes != key_shapes
        ):
            wihG, whhG = self.custom_ops_lib.gru_bi_prepack_gate_major_f16_cuda(w_ih, w_hh)
            self._packed_wihG = wihG
            self._packed_whhG = whhG
            self._wih_ptrs = wih_ptrs
            self._whh_ptrs = whh_ptrs
            self._packed_shapes = key_shapes

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        can_custom = (
            x.is_cuda and h0.is_cuda and
            x.dtype == torch.float16 and h0.dtype == torch.float16 and
            (not self.batch_first) and
            self.bias and
            self.gru.bidirectional and
            self.num_layers == 1 and
            x.dim() == 3 and h0.dim() == 3 and
            h0.size(0) == 2 and h0.size(2) == self.hidden_size and
            x.size(2) == self.input_size
        )
        if not can_custom:
            _out, h_n = self.gru(x, h0)
            return h_n

        self._maybe_repack()

        # fuse biases on host side: bsum = b_ih + b_hh, shape (2,3H) then view as (2,3,H)
        b_ih = torch.stack([self.gru.bias_ih_l0, self.gru.bias_ih_l0_reverse], dim=0).contiguous()
        b_hh = torch.stack([self.gru.bias_hh_l0, self.gru.bias_hh_l0_reverse], dim=0).contiguous()
        bsum = (b_ih + b_hh).contiguous().view(2, 3, self.hidden_size)

        hn = self.custom_ops_lib.gru_bi_hidden_single_layer_f16_warp_cuda(
            x.contiguous(),
            h0.contiguous(),
            self._packed_wihG,
            self._packed_whhG,
            bsum,
        )
        return hn