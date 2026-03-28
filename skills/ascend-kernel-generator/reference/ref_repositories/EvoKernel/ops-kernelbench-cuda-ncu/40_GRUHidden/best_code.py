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

static __device__ __forceinline__ float2 half2_to_float2(const half2 &h2) {
    float2 f;
    f.x = __half2float(__low2half(h2));
    f.y = __half2float(__high2half(h2));
    return f;
}

__global__ void transpose_f16_kernel(const half* __restrict__ in, half* __restrict__ out, int M, int N) {
    int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int m = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (m < M && n < N) out[n * M + m] = in[m * N + n];
}

// Scalar fallback (close to baseline v3): one block per batch element, ping-pong shared hidden
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void gru_hidden_fused_f16_kernel_scalar(
    const half* __restrict__ x,     // (T,B,I)
    const half* __restrict__ h0,    // (1,B,H)
    const half* __restrict__ wihT,  // (I,3H)
    const half* __restrict__ whhT,  // (H,3H)
    const half* __restrict__ b,     // (3H)
    half* __restrict__ hn,          // (1,B,H)
    int T, int B, int I, int H
) {
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    extern __shared__ half shmem[];
    half* sh_x  = shmem;          // I
    half* sh_h0 = sh_x + I;       // H
    half* sh_h1 = sh_h0 + H;      // H

    for (int h = (int)threadIdx.x; h < H; h += THREADS) {
        sh_h0[h] = h0[bidx * H + h];
    }
    __syncthreads();

    const int stride3H = 3 * H;
    half* sh_h_cur = sh_h0;
    half* sh_h_nxt = sh_h1;

    for (int t = 0; t < T; ++t) {
        const half* x_ptr = x + (t * B + bidx) * I;
        for (int i = (int)threadIdx.x; i < I; i += THREADS) sh_x[i] = x_ptr[i];
        __syncthreads();

        for (int h = (int)threadIdx.x; h < H; h += THREADS) {
            float ir = __half2float(__ldg(b + (0 * H + h)));
            float iz = __half2float(__ldg(b + (1 * H + h)));
            float in_ = __half2float(__ldg(b + (2 * H + h)));

            float hr = 0.f, hz = 0.f, hn_ = 0.f;

            #pragma unroll 1
            for (int k = 0; k < I; ++k) {
                float xv = __half2float(sh_x[k]);
                const half* w = wihT + k * stride3H;
                ir = fmaf(xv, __half2float(__ldg(w + (0 * H + h))), ir);
                iz = fmaf(xv, __half2float(__ldg(w + (1 * H + h))), iz);
                in_ = fmaf(xv, __half2float(__ldg(w + (2 * H + h))), in_);
            }

            #pragma unroll 1
            for (int k = 0; k < H; ++k) {
                float hv = __half2float(sh_h_cur[k]);
                const half* w = whhT + k * stride3H;
                hr = fmaf(hv, __half2float(__ldg(w + (0 * H + h))), hr);
                hz = fmaf(hv, __half2float(__ldg(w + (1 * H + h))), hz);
                hn_ = fmaf(hv, __half2float(__ldg(w + (2 * H + h))), hn_);
            }

            float r = sigmoidf_fast(ir + hr);
            float z = sigmoidf_fast(iz + hz);
            float nval = tanhf_fast(in_ + r * hn_);

            float hprev = __half2float(sh_h_cur[h]);
            float hnew = (1.0f - z) * nval + z * hprev;
            sh_h_nxt[h] = __float2half_rn(hnew);
        }

        __syncthreads();
        half* tmp = sh_h_cur; sh_h_cur = sh_h_nxt; sh_h_nxt = tmp;
    }

    for (int h = (int)threadIdx.x; h < H; h += THREADS) {
        hn[bidx * H + h] = sh_h_cur[h];
    }
}

// Optimized half2 kernel: one block per batch element, shared x and shared h,
// compute two hidden units per thread (H even). Update h in-place after barrier.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 3)
void gru_hidden_fused_f16_kernel_h2(
    const half* __restrict__ x,     // (T,B,I)
    const half* __restrict__ h0,    // (1,B,H)
    const half* __restrict__ wihT,  // (I,3H)
    const half* __restrict__ whhT,  // (H,3H)
    const half* __restrict__ b,     // (3H)
    half* __restrict__ hn,          // (1,B,H)
    int T, int B, int I, int H
) {
    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    extern __shared__ half shmem[];
    half* sh_x = shmem;       // I
    half* sh_h = sh_x + I;    // H

    // init hidden
    for (int h = (int)threadIdx.x; h < H; h += THREADS) {
        sh_h[h] = h0[bidx * H + h];
    }
    __syncthreads();

    const int stride3H = 3 * H;
    const int H2 = H >> 1;
    const int I2 = I >> 1;

    // thread owns one half2 hidden column: h2_idx in [0, H2)
    for (int t = 0; t < T; ++t) {
        // load x_t into shared (vectorized when possible)
        const half* x_ptr = x + (t * B + bidx) * I;
        if ((I & 1) == 0) {
            half2* shx2 = reinterpret_cast<half2*>(sh_x);
            const half2* xp2 = reinterpret_cast<const half2*>(x_ptr);
            for (int i2 = (int)threadIdx.x; i2 < I2; i2 += THREADS) shx2[i2] = xp2[i2];
        } else {
            for (int i = (int)threadIdx.x; i < I; i += THREADS) sh_x[i] = x_ptr[i];
        }
        __syncthreads();

        int tid = (int)threadIdx.x;
        int h2_idx = tid;  // one half2 per thread (only first H2 threads active)
        half2 hnew2 = __float2half2_rn(0.f);

        if (h2_idx < H2) {
            int hcol = h2_idx << 1;

            // bias load as half2 (contiguous)
            const half2* b2 = reinterpret_cast<const half2*>(b);
            half2 bir2 = __ldg(b2 + (0 * H2 + h2_idx));
            half2 biz2 = __ldg(b2 + (1 * H2 + h2_idx));
            half2 bin2 = __ldg(b2 + (2 * H2 + h2_idx));

            float2 ir = half2_to_float2(bir2);
            float2 iz = half2_to_float2(biz2);
            float2 in_ = half2_to_float2(bin2);

            float2 hr = {0.f, 0.f};
            float2 hz = {0.f, 0.f};
            float2 hn_ = {0.f, 0.f};

            // input dot: stream over I, load two weights (h,h+1) for each gate
            if ((I & 1) == 0) {
                const half2* shx2 = reinterpret_cast<const half2*>(sh_x);
                #pragma unroll 1
                for (int k2 = 0; k2 < I2; ++k2) {
                    half2 xv2 = shx2[k2];
                    float2 xv = half2_to_float2(xv2);
                    int k0 = k2 << 1;

                    const half* w0 = wihT + (k0 + 0) * stride3H + hcol;
                    const half* w1 = wihT + (k0 + 1) * stride3H + hcol;

                    half2 wr0 = __ldg(reinterpret_cast<const half2*>(w0 + 0 * H));
                    half2 wz0 = __ldg(reinterpret_cast<const half2*>(w0 + 1 * H));
                    half2 wn0 = __ldg(reinterpret_cast<const half2*>(w0 + 2 * H));
                    half2 wr1 = __ldg(reinterpret_cast<const half2*>(w1 + 0 * H));
                    half2 wz1 = __ldg(reinterpret_cast<const half2*>(w1 + 1 * H));
                    half2 wn1 = __ldg(reinterpret_cast<const half2*>(w1 + 2 * H));

                    float2 fwr0 = half2_to_float2(wr0), fwz0 = half2_to_float2(wz0), fwn0 = half2_to_float2(wn0);
                    float2 fwr1 = half2_to_float2(wr1), fwz1 = half2_to_float2(wz1), fwn1 = half2_to_float2(wn1);

                    ir.x = fmaf(xv.x, fwr0.x, ir.x); ir.y = fmaf(xv.x, fwr0.y, ir.y);
                    iz.x = fmaf(xv.x, fwz0.x, iz.x); iz.y = fmaf(xv.x, fwz0.y, iz.y);
                    in_.x = fmaf(xv.x, fwn0.x, in_.x); in_.y = fmaf(xv.x, fwn0.y, in_.y);

                    ir.x = fmaf(xv.y, fwr1.x, ir.x); ir.y = fmaf(xv.y, fwr1.y, ir.y);
                    iz.x = fmaf(xv.y, fwz1.x, iz.x); iz.y = fmaf(xv.y, fwz1.y, iz.y);
                    in_.x = fmaf(xv.y, fwn1.x, in_.x); in_.y = fmaf(xv.y, fwn1.y, in_.y);
                }
            } else {
                #pragma unroll 1
                for (int k = 0; k < I; ++k) {
                    float xv = __half2float(sh_x[k]);
                    const half* w = wihT + k * stride3H + hcol;
                    half2 wr = __ldg(reinterpret_cast<const half2*>(w + 0 * H));
                    half2 wz = __ldg(reinterpret_cast<const half2*>(w + 1 * H));
                    half2 wn = __ldg(reinterpret_cast<const half2*>(w + 2 * H));
                    float2 fwr = half2_to_float2(wr), fwz = half2_to_float2(wz), fwn = half2_to_float2(wn);
                    ir.x = fmaf(xv, fwr.x, ir.x); ir.y = fmaf(xv, fwr.y, ir.y);
                    iz.x = fmaf(xv, fwz.x, iz.x); iz.y = fmaf(xv, fwz.y, iz.y);
                    in_.x = fmaf(xv, fwn.x, in_.x); in_.y = fmaf(xv, fwn.y, in_.y);
                }
            }

            // recurrent dot: stream over H, using shared h scalar (for simplicity) but vectorized weights
            #pragma unroll 1
            for (int k = 0; k < H; ++k) {
                float hv = __half2float(sh_h[k]);
                const half* w = whhT + k * stride3H + hcol;
                half2 wr = __ldg(reinterpret_cast<const half2*>(w + 0 * H));
                half2 wz = __ldg(reinterpret_cast<const half2*>(w + 1 * H));
                half2 wn = __ldg(reinterpret_cast<const half2*>(w + 2 * H));
                float2 fwr = half2_to_float2(wr), fwz = half2_to_float2(wz), fwn = half2_to_float2(wn);
                hr.x = fmaf(hv, fwr.x, hr.x); hr.y = fmaf(hv, fwr.y, hr.y);
                hz.x = fmaf(hv, fwz.x, hz.x); hz.y = fmaf(hv, fwz.y, hz.y);
                hn_.x = fmaf(hv, fwn.x, hn_.x); hn_.y = fmaf(hv, fwn.y, hn_.y);
            }

            float r0 = sigmoidf_fast(ir.x + hr.x);
            float r1 = sigmoidf_fast(ir.y + hr.y);
            float z0 = sigmoidf_fast(iz.x + hz.x);
            float z1 = sigmoidf_fast(iz.y + hz.y);

            float n0 = tanhf_fast(in_.x + r0 * hn_.x);
            float n1 = tanhf_fast(in_.y + r1 * hn_.y);

            float hprev0 = __half2float(sh_h[hcol + 0]);
            float hprev1 = __half2float(sh_h[hcol + 1]);

            float hnew0 = (1.0f - z0) * n0 + z0 * hprev0;
            float hnew1 = (1.0f - z1) * n1 + z1 * hprev1;

            hnew2 = __floats2half2_rn(hnew0, hnew1);
        }

        __syncthreads();
        // in-place update after all threads finished reading old sh_h
        if (h2_idx < H2) {
            reinterpret_cast<half2*>(sh_h)[h2_idx] = hnew2;
        }
        __syncthreads();
    }

    // store final
    for (int h = (int)threadIdx.x; h < H; h += THREADS) {
        hn[bidx * H + h] = sh_h[h];
    }
}

torch::Tensor gru_hidden_forward_single_layer_f16_cuda_prepacked_v5(
    torch::Tensor x,      // (T,B,I) half
    torch::Tensor h0,     // (1,B,H) half
    torch::Tensor wihT,   // (I,3H) half
    torch::Tensor whhT,   // (H,3H) half
    torch::Tensor b_fused // (3H) half
) {
    TORCH_CHECK(x.is_cuda() && h0.is_cuda(), "x/h0 must be CUDA");
    TORCH_CHECK(wihT.is_cuda() && whhT.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(b_fused.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16 && h0.dtype() == torch::kFloat16, "x/h0 must be float16");
    TORCH_CHECK(wihT.dtype() == torch::kFloat16 && whhT.dtype() == torch::kFloat16, "weights must be float16");
    TORCH_CHECK(b_fused.dtype() == torch::kFloat16, "bias must be float16");

    TORCH_CHECK(x.dim() == 3, "x must be (T,B,I)");
    TORCH_CHECK(h0.dim() == 3, "h0 must be (1,B,H)");
    TORCH_CHECK(wihT.dim() == 2 && whhT.dim() == 2, "weights must be 2D");
    TORCH_CHECK(b_fused.dim() == 1, "bias must be 1D");

    auto xc = x.contiguous();
    auto h0c = h0.contiguous();
    auto wihTc = wihT.contiguous();
    auto whhTc = whhT.contiguous();
    auto bc = b_fused.contiguous();

    int T = (int)xc.size(0);
    int B = (int)xc.size(1);
    int I = (int)xc.size(2);
    TORCH_CHECK((int)h0c.size(0) == 1, "h0 first dim must be 1");
    TORCH_CHECK((int)h0c.size(1) == B, "h0 batch mismatch");
    int H = (int)h0c.size(2);

    TORCH_CHECK((int)wihTc.size(0) == I && (int)wihTc.size(1) == 3 * H, "wihT must be (I,3H)");
    TORCH_CHECK((int)whhTc.size(0) == H && (int)whhTc.size(1) == 3 * H, "whhT must be (H,3H)");
    TORCH_CHECK((int)bc.numel() == 3 * H, "b_fused must be (3H)");

    auto hn = torch::empty({1, B, H}, xc.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // For h2 kernel we need H even (half2 across hidden) and we prefer THREADS=128.
    // Shared memory: sh_x(I) + sh_h(H)
    constexpr int THREADS_OPT = 128;
    constexpr int THREADS_FBK = 256;

    size_t shmem_opt = (size_t)(I + H) * sizeof(half);
    size_t shmem_fbk = (size_t)(I + 2 * H) * sizeof(half);

    dim3 grid(B);

    bool can_h2 = ((H & 1) == 0) && (shmem_opt <= 96 * 1024);
    if (can_h2) {
        dim3 block(THREADS_OPT);
        gru_hidden_fused_f16_kernel_h2<THREADS_OPT><<<grid, block, shmem_opt, stream>>>(
            (const half*)xc.data_ptr<at::Half>(),
            (const half*)h0c.data_ptr<at::Half>(),
            (const half*)wihTc.data_ptr<at::Half>(),
            (const half*)whhTc.data_ptr<at::Half>(),
            (const half*)bc.data_ptr<at::Half>(),
            (half*)hn.data_ptr<at::Half>(),
            T, B, I, H
        );
    } else {
        TORCH_CHECK(shmem_fbk <= 96 * 1024, "shared memory too large for scalar fallback");
        dim3 block(THREADS_FBK);
        gru_hidden_fused_f16_kernel_scalar<THREADS_FBK><<<grid, block, shmem_fbk, stream>>>(
            (const half*)xc.data_ptr<at::Half>(),
            (const half*)h0c.data_ptr<at::Half>(),
            (const half*)wihTc.data_ptr<at::Half>(),
            (const half*)whhTc.data_ptr<at::Half>(),
            (const half*)bc.data_ptr<at::Half>(),
            (half*)hn.data_ptr<at::Half>(),
            T, B, I, H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return hn;
}

torch::Tensor prepack_w_transpose_f16_cuda(torch::Tensor w) {
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(w.dtype() == torch::kFloat16, "w must be float16");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    auto wc = w.contiguous();
    int M = (int)wc.size(0);
    int N = (int)wc.size(1);
    auto wT = torch::empty({N, M}, wc.options());

    dim3 blk(32, 8);
    dim3 grd((N + blk.x - 1) / blk.x, (M + blk.y - 1) / blk.y);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    transpose_f16_kernel<<<grd, blk, 0, stream>>>(
        (const half*)wc.data_ptr<at::Half>(),
        (half*)wT.data_ptr<at::Half>(),
        M, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return wT;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor gru_hidden_forward_single_layer_f16_cuda_prepacked_v5(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor wihT,
    torch::Tensor whhT,
    torch::Tensor b_fused
);

torch::Tensor prepack_w_transpose_f16_cuda(torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gru_hidden_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "gru_hidden_forward_single_layer_f16_cuda_prepacked_v5",
        "prepack_w_transpose_f16_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Operator replacement for `gru_hidden` (returns h_n only).

    Custom CUDA path supports ONLY:
      - CUDA float16
      - batch_first=False
      - bias=True
      - bidirectional=False
      - dropout=0
      - num_layers=1
      - returns h_n with shape (1,B,H)

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
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=0.0,
            bidirectional=False,
        ).half()

        self.custom_ops_lib = custom_ops_lib

        self._wihT = None
        self._whhT = None
        self._wih_storage_id = None
        self._whh_storage_id = None
        self._packed_device = None
        self._packed_dtype = None

        self._b_fused = None
        self._b_storage_id = None

    def _storage_id(self, t: torch.Tensor) -> int:
        return int(t.untyped_storage().data_ptr())

    def _maybe_prepack(self):
        w_ih = self.gru.weight_ih_l0
        w_hh = self.gru.weight_hh_l0

        dev = w_ih.device
        dt = w_ih.dtype
        sid_ih = self._storage_id(w_ih)
        sid_hh = self._storage_id(w_hh)

        need_w = (
            self._wihT is None or self._whhT is None or
            self._wih_storage_id != sid_ih or
            self._whh_storage_id != sid_hh or
            self._packed_device != dev or
            self._packed_dtype != dt
        )
        if need_w:
            self._wihT = self.custom_ops_lib.prepack_w_transpose_f16_cuda(w_ih)  # (I,3H)
            self._whhT = self.custom_ops_lib.prepack_w_transpose_f16_cuda(w_hh)  # (H,3H)
            self._wih_storage_id = sid_ih
            self._whh_storage_id = sid_hh
            self._packed_device = dev
            self._packed_dtype = dt

        b_ih = self.gru.bias_ih_l0
        b_hh = self.gru.bias_hh_l0
        sid_b = (self._storage_id(b_ih), self._storage_id(b_hh))
        need_b = (
            self._b_fused is None or
            self._b_storage_id != sid_b or
            self._packed_device != dev or
            self._packed_dtype != dt
        )
        if need_b:
            self._b_fused = (b_ih + b_hh).contiguous()
            self._b_storage_id = sid_b

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        can_custom = (
            x.is_cuda and h0.is_cuda and
            x.dtype == torch.float16 and h0.dtype == torch.float16 and
            (not self.batch_first) and
            self.bias and
            (not self.gru.bidirectional) and
            self.num_layers == 1 and
            x.dim() == 3 and h0.dim() == 3 and
            h0.size(0) == 1 and
            x.size(2) == self.input_size and
            h0.size(2) == self.hidden_size
        )
        if not can_custom:
            _out, hn = self.gru(x, h0)
            return hn

        self._maybe_prepack()

        # Flatten (1,B,H) -> (B,H) inside kernel interface for slightly less index math
        h0_flat = h0.contiguous().view(-1, h0.size(2))
        hn_flat = self.custom_ops_lib.gru_hidden_forward_single_layer_f16_cuda_prepacked_v5(
            x.contiguous(),
            h0_flat.unsqueeze(0),  # keep signature (1,B,H)
            self._wihT,
            self._whhT,
            self._b_fused,
        )
        return hn_flat