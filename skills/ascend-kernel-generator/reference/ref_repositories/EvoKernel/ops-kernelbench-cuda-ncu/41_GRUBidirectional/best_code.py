import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG(x) __ldg(x)
#else
#define LDG(x) (*(x))
#endif

static __device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
static __device__ __forceinline__ float tanhf_fast(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ half2 h2ldg(const half2* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float2 h2_to_f2(half2 v) {
    float2 f;
    f.x = __half2float(__low2half(v));
    f.y = __half2float(__high2half(v));
    return f;
}

static __device__ __forceinline__ half2 f2_to_h2(float a, float b) {
    return __halves2half2(__float2half_rn(a), __float2half_rn(b));
}

// Pack weights from (3H, K) into (K, H, 3) with gate-last contiguous.
// For w_ih: K=I. For w_hh: K=H.
// Input layout: gate-major [r(0..H-1), z(H..2H-1), n(2H..3H-1)].
__global__ void pack_w_gatelast_f16_kernel(
    const half* __restrict__ w_in,  // (3H, K) row-major contiguous
    half* __restrict__ w_out,       // (K, H, 3) contiguous
    int K, int H
){
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int h = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (k >= K || h >= H) return;

    const int strideK = K;
    half wr = w_in[(0 * H + h) * strideK + k];
    half wz = w_in[(1 * H + h) * strideK + k];
    half wn = w_in[(2 * H + h) * strideK + k];

    // out index: ((k*H + h)*3 + g)
    int base = (k * H + h) * 3;
    w_out[base + 0] = wr;
    w_out[base + 1] = wz;
    w_out[base + 2] = wn;
}

// Fused one-direction GRU forward using packed weights (K,H,3).
// One CTA per batch element, warps tile over hidden dimension, computing 2 hidden units at once (half2).
template<int THREADS>
__global__ __launch_bounds__(THREADS, 3)
void gru_fwd_dir_packed_gatelast_f16_kernel(
    const half* __restrict__ x,        // (T,B,I)
    const half* __restrict__ h0,       // (2,B,H)
    const half* __restrict__ wih_p,    // (2, I, H, 3) contiguous
    const half* __restrict__ whh_p,    // (2, H, H, 3) contiguous
    const half* __restrict__ b_ih,     // (2,3H)
    const half* __restrict__ b_hh,     // (2,3H)
    half* __restrict__ y,              // (T,B,2H)
    half* __restrict__ hn,             // (2,B,H)
    int T, int B, int I, int H,
    int dir
){
    int b = (int)blockIdx.x;
    if (b >= B) return;

    extern __shared__ half shmem[];
    half* sh_x  = shmem;          // I
    half* sh_h0 = sh_x + I;       // H
    half* sh_h1 = sh_h0 + H;      // H

    // init hidden
    const half* h0_ptr = h0 + (dir * B + b) * H;
    for (int i = (int)threadIdx.x; i < H; i += THREADS) sh_h0[i] = LDG(h0_ptr + i);
    __syncthreads();

    const half* wih_dir = wih_p + (size_t)dir * (size_t)I * (size_t)H * 3;
    const half* whh_dir = whh_p + (size_t)dir * (size_t)H * (size_t)H * 3;
    const half* bih_dir = b_ih + dir * (3 * H);
    const half* bhh_dir = b_hh + dir * (3 * H);

    half* sh_prev = sh_h0;
    half* sh_next = sh_h1;

    // hidden in half2 domain when possible
    int H2 = H >> 1;

    for (int t = 0; t < T; ++t) {
        int tt = (dir == 0) ? t : (T - 1 - t);

        // stage x_t into shared
        const half* x_ptr = x + (tt * B + b) * I;
        for (int i = (int)threadIdx.x; i < I; i += THREADS) sh_x[i] = LDG(x_ptr + i);
        __syncthreads();

        // compute hidden
        if ((H & 1) == 0) {
            // process two hidden units at once
            int tid = (int)threadIdx.x;
            for (int h2 = tid; h2 < H2; h2 += THREADS) {
                int h = h2 << 1;

                // bias (gate-major)
                float2 bir = h2_to_f2(*(const half2*)(bih_dir + 0 * H + h));
                float2 biz = h2_to_f2(*(const half2*)(bih_dir + 1 * H + h));
                float2 bin = h2_to_f2(*(const half2*)(bih_dir + 2 * H + h));

                float2 bhr = h2_to_f2(*(const half2*)(bhh_dir + 0 * H + h));
                float2 bhz = h2_to_f2(*(const half2*)(bhh_dir + 1 * H + h));
                float2 bhn = h2_to_f2(*(const half2*)(bhh_dir + 2 * H + h));

                float2 ir = bir, iz = biz, in_ = bin;
                float2 hr = bhr, hz = bhz, hn_ = bhn;

                // input dot: sum_k x[k] * Wih[k, h, g]
                #pragma unroll 1
                for (int k = 0; k < I; ++k) {
                    float xv = __half2float(sh_x[k]);
                    // w base: (k*H + h)*3
                    const half* wbase = wih_dir + ((k * H + h) * 3);
                    half2 wr = *(const half2*)(wbase + 0); // loads (wr[h], wr[h+1]) because gate-last => [wr0,wz0,wn0, wr1,wz1,wn1] is NOT contiguous.
                    // Since gate-last is (..,3), two h are separated by 3, so cannot load wr as half2 directly.
                    // We instead load scalars for each h lane (still fewer address calcs for 3 gates).
                    float wr0 = __half2float(LDG(wbase + 0));
                    float wz0 = __half2float(LDG(wbase + 1));
                    float wn0 = __half2float(LDG(wbase + 2));
                    const half* wbase1 = wbase + 3;
                    float wr1 = __half2float(LDG(wbase1 + 0));
                    float wz1 = __half2float(LDG(wbase1 + 1));
                    float wn1 = __half2float(LDG(wbase1 + 2));

                    ir.x += xv * wr0; iz.x += xv * wz0; in_.x += xv * wn0;
                    ir.y += xv * wr1; iz.y += xv * wz1; in_.y += xv * wn1;
                }

                // recurrent dot: sum_k hprev[k] * Whh[k, h, g]
                const half2* hprev2 = (const half2*)sh_prev;
                #pragma unroll 1
                for (int k2 = 0; k2 < H2; ++k2) {
                    half2 hv2 = hprev2[k2];
                    float2 hv = h2_to_f2(hv2);
                    int k = k2 << 1;

                    const half* wbase0 = whh_dir + ((k * H + h) * 3);
                    float wr00 = __half2float(LDG(wbase0 + 0));
                    float wz00 = __half2float(LDG(wbase0 + 1));
                    float wn00 = __half2float(LDG(wbase0 + 2));
                    const half* wbase0_1 = wbase0 + 3;
                    float wr01 = __half2float(LDG(wbase0_1 + 0));
                    float wz01 = __half2float(LDG(wbase0_1 + 1));
                    float wn01 = __half2float(LDG(wbase0_1 + 2));

                    hr.x += hv.x * wr00; hz.x += hv.x * wz00; hn_.x += hv.x * wn00;
                    hr.y += hv.x * wr01; hz.y += hv.x * wz01; hn_.y += hv.x * wn01;

                    const half* wbase1 = whh_dir + (((k + 1) * H + h) * 3);
                    float wr10 = __half2float(LDG(wbase1 + 0));
                    float wz10 = __half2float(LDG(wbase1 + 1));
                    float wn10 = __half2float(LDG(wbase1 + 2));
                    const half* wbase1_1 = wbase1 + 3;
                    float wr11 = __half2float(LDG(wbase1_1 + 0));
                    float wz11 = __half2float(LDG(wbase1_1 + 1));
                    float wn11 = __half2float(LDG(wbase1_1 + 2));

                    hr.x += hv.y * wr10; hz.x += hv.y * wz10; hn_.x += hv.y * wn10;
                    hr.y += hv.y * wr11; hz.y += hv.y * wz11; hn_.y += hv.y * wn11;
                }

                float r0 = sigmoidf_fast(ir.x + hr.x);
                float z0 = sigmoidf_fast(iz.x + hz.x);
                float n0 = tanhf_fast(in_.x + r0 * hn_.x);

                float r1 = sigmoidf_fast(ir.y + hr.y);
                float z1 = sigmoidf_fast(iz.y + hz.y);
                float n1 = tanhf_fast(in_.y + r1 * hn_.y);

                float2 hp = h2_to_f2(*(const half2*)(sh_prev + h));
                float hnew0 = (1.0f - z0) * n0 + z0 * hp.x;
                float hnew1 = (1.0f - z1) * n1 + z1 * hp.y;

                *(half2*)(sh_next + h) = f2_to_h2(hnew0, hnew1);
                *(half2*)(y + (size_t)(t * B + b) * (size_t)(2 * H) + (size_t)dir * (size_t)H + h) =
                    f2_to_h2(hnew0, hnew1);
            }
        } else {
            // fallback scalar H
            for (int h = (int)threadIdx.x; h < H; h += THREADS) {
                float ir = __half2float(LDG(bih_dir + (0 * H + h)));
                float iz = __half2float(LDG(bih_dir + (1 * H + h)));
                float in_ = __half2float(LDG(bih_dir + (2 * H + h)));

                float hr = __half2float(LDG(bhh_dir + (0 * H + h)));
                float hz = __half2float(LDG(bhh_dir + (1 * H + h)));
                float hn_ = __half2float(LDG(bhh_dir + (2 * H + h)));

                #pragma unroll 1
                for (int k = 0; k < I; ++k) {
                    float xv = __half2float(sh_x[k]);
                    const half* wbase = wih_dir + ((k * H + h) * 3);
                    ir += xv * __half2float(LDG(wbase + 0));
                    iz += xv * __half2float(LDG(wbase + 1));
                    in_ += xv * __half2float(LDG(wbase + 2));
                }

                #pragma unroll 1
                for (int k = 0; k < H; ++k) {
                    float hv = __half2float(sh_prev[k]);
                    const half* wbase = whh_dir + ((k * H + h) * 3);
                    hr += hv * __half2float(LDG(wbase + 0));
                    hz += hv * __half2float(LDG(wbase + 1));
                    hn_ += hv * __half2float(LDG(wbase + 2));
                }

                float r = sigmoidf_fast(ir + hr);
                float z = sigmoidf_fast(iz + hz);
                float nval = tanhf_fast(in_ + r * hn_);
                float hprev = __half2float(sh_prev[h]);
                float hnew = (1.0f - z) * nval + z * hprev;

                sh_next[h] = __float2half_rn(hnew);
                y[(size_t)(t * B + b) * (size_t)(2 * H) + (size_t)dir * (size_t)H + h] = __float2half_rn(hnew);
            }
        }

        __syncthreads();
        half* tmp = sh_prev; sh_prev = sh_next; sh_next = tmp;
    }

    for (int h = (int)threadIdx.x; h < H; h += THREADS) {
        hn[(dir * B + b) * H + h] = sh_prev[h];
    }
}

std::vector<torch::Tensor> gru_bi_forward_single_layer_f16_packed_gatelast_cuda(
    torch::Tensor x,    // (T,B,I) half
    torch::Tensor h0,   // (2,B,H) half
    torch::Tensor wih_p,// (2,I,H,3) half
    torch::Tensor whh_p,// (2,H,H,3) half
    torch::Tensor b_ih, // (2,3H) half
    torch::Tensor b_hh  // (2,3H) half
){
    TORCH_CHECK(x.is_cuda() && h0.is_cuda(), "x/h0 must be CUDA");
    TORCH_CHECK(wih_p.is_cuda() && whh_p.is_cuda(), "packed weights must be CUDA");
    TORCH_CHECK(b_ih.is_cuda() && b_hh.is_cuda(), "biases must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16 && h0.dtype() == torch::kFloat16, "x/h0 must be float16");
    TORCH_CHECK(wih_p.dtype() == torch::kFloat16 && whh_p.dtype() == torch::kFloat16, "packed weights must be float16");
    TORCH_CHECK(b_ih.dtype() == torch::kFloat16 && b_hh.dtype() == torch::kFloat16, "biases must be float16");

    auto xc = x.contiguous();
    auto h0c = h0.contiguous();
    auto wihc = wih_p.contiguous();
    auto whhc = whh_p.contiguous();
    auto bihc = b_ih.contiguous();
    auto bhhc = b_hh.contiguous();

    TORCH_CHECK(xc.dim() == 3, "x must be (T,B,I)");
    TORCH_CHECK(h0c.dim() == 3 && (int)h0c.size(0) == 2, "h0 must be (2,B,H)");
    TORCH_CHECK(wihc.dim() == 4 && (int)wihc.size(0) == 2 && (int)wihc.size(3) == 3, "wih_p must be (2,I,H,3)");
    TORCH_CHECK(whhc.dim() == 4 && (int)whhc.size(0) == 2 && (int)whhc.size(3) == 3, "whh_p must be (2,H,H,3)");
    TORCH_CHECK(bihc.dim() == 2 && bhhc.dim() == 2, "biases must be (2,3H)");

    int T = (int)xc.size(0);
    int B = (int)xc.size(1);
    int I = (int)xc.size(2);
    int H = (int)h0c.size(2);

    TORCH_CHECK((int)h0c.size(1) == B, "h0 shape mismatch");
    TORCH_CHECK((int)wihc.size(1) == I && (int)wihc.size(2) == H, "wih_p shape mismatch");
    TORCH_CHECK((int)whhc.size(1) == H && (int)whhc.size(2) == H, "whh_p shape mismatch");
    TORCH_CHECK((int)bihc.size(0) == 2 && (int)bihc.size(1) == 3 * H, "b_ih shape mismatch");
    TORCH_CHECK((int)bhhc.size(0) == 2 && (int)bhhc.size(1) == 3 * H, "b_hh shape mismatch");

    auto y  = torch::empty({T, B, 2 * H}, xc.options());
    auto hn = torch::empty({2, B, H}, xc.options());

    constexpr int THREADS = 128;
    dim3 grid((unsigned)B);
    dim3 block(THREADS);

    size_t shmem = (size_t)(I + 2 * H) * sizeof(half);
    TORCH_CHECK(shmem <= 96 * 1024, "shared memory too large for (I+2H)");

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    for (int dir = 0; dir < 2; ++dir) {
        gru_fwd_dir_packed_gatelast_f16_kernel<THREADS><<<grid, block, shmem, stream>>>(
            (const half*)xc.data_ptr<at::Half>(),
            (const half*)h0c.data_ptr<at::Half>(),
            (const half*)wihc.data_ptr<at::Half>(),
            (const half*)whhc.data_ptr<at::Half>(),
            (const half*)bihc.data_ptr<at::Half>(),
            (const half*)bhhc.data_ptr<at::Half>(),
            (half*)y.data_ptr<at::Half>(),
            (half*)hn.data_ptr<at::Half>(),
            T, B, I, H, dir
        );
    }

    return {y, hn};
}

std::vector<torch::Tensor> gru_bi_prepack_gatelast_f16_cuda(
    torch::Tensor w_ih, // (2,3H,I)
    torch::Tensor w_hh  // (2,3H,H)
){
    TORCH_CHECK(w_ih.is_cuda() && w_hh.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(w_ih.dtype() == torch::kFloat16 && w_hh.dtype() == torch::kFloat16, "weights must be float16");
    auto wih = w_ih.contiguous();
    auto whh = w_hh.contiguous();
    TORCH_CHECK(wih.dim() == 3 && whh.dim() == 3, "weights must be 3D");
    int D = (int)wih.size(0);
    TORCH_CHECK(D == 2, "expected 2 directions");
    int threeH = (int)wih.size(1);
    int I = (int)wih.size(2);
    TORCH_CHECK((int)whh.size(1) == threeH, "w_hh mismatch");
    int H = (int)whh.size(2);
    TORCH_CHECK(threeH == 3 * H, "3H mismatch");

    auto opts = wih.options();
    auto wih_p = torch::empty({2, I, H, 3}, opts);
    auto whh_p = torch::empty({2, H, H, 3}, opts);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(32, 8);

    for (int dir = 0; dir < 2; ++dir) {
        const half* wih_in = (const half*)wih.data_ptr<at::Half>() + (size_t)dir * (size_t)(3 * H) * (size_t)I;
        half* wih_out = (half*)wih_p.data_ptr<at::Half>() + (size_t)dir * (size_t)I * (size_t)H * 3;
        dim3 grid_wih((I + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        pack_w_gatelast_f16_kernel<<<grid_wih, block, 0, stream>>>(wih_in, wih_out, I, H);

        const half* whh_in = (const half*)whh.data_ptr<at::Half>() + (size_t)dir * (size_t)(3 * H) * (size_t)H;
        half* whh_out = (half*)whh_p.data_ptr<at::Half>() + (size_t)dir * (size_t)H * (size_t)H * 3;
        dim3 grid_whh((H + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        pack_w_gatelast_f16_kernel<<<grid_whh, block, 0, stream>>>(whh_in, whh_out, H, H);
    }

    return {wih_p, whh_p};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> gru_bi_forward_single_layer_f16_packed_gatelast_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor wih_p,
    torch::Tensor whh_p,
    torch::Tensor b_ih,
    torch::Tensor b_hh
);

std::vector<torch::Tensor> gru_bi_prepack_gatelast_f16_cuda(
    torch::Tensor w_ih,
    torch::Tensor w_hh
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gru_bi_fused_v5_gatelast",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "gru_bi_forward_single_layer_f16_packed_gatelast_cuda",
        "gru_bi_prepack_gatelast_f16_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized bidirectional single-layer GRU forward (returns output only).

    Custom CUDA path supports ONLY:
      - CUDA float16
      - batch_first=False
      - bias=True
      - dropout=0
      - num_layers=1
      - bidirectional=True
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
            bidirectional=True,
        ).half()

        self.custom_ops_lib = custom_ops_lib

        self.register_buffer("_wih_p", None, persistent=False)
        self.register_buffer("_whh_p", None, persistent=False)
        self.register_buffer("_bih", None, persistent=False)
        self.register_buffer("_bhh", None, persistent=False)
        self._cache_key = None

    @torch.no_grad()
    def _maybe_prepack(self, x: torch.Tensor):
        # Invalidate if weight storage changes (data_ptr) or device changes.
        key = (x.device, x.dtype,
               self.gru.weight_ih_l0.data_ptr(),
               self.gru.weight_hh_l0.data_ptr(),
               self.gru.weight_ih_l0_reverse.data_ptr(),
               self.gru.weight_hh_l0_reverse.data_ptr())
        if self._cache_key == key and self._wih_p is not None:
            return

        w_ih = torch.stack([self.gru.weight_ih_l0, self.gru.weight_ih_l0_reverse], dim=0).contiguous()
        w_hh = torch.stack([self.gru.weight_hh_l0, self.gru.weight_hh_l0_reverse], dim=0).contiguous()
        b_ih = torch.stack([self.gru.bias_ih_l0, self.gru.bias_ih_l0_reverse], dim=0).contiguous()
        b_hh = torch.stack([self.gru.bias_hh_l0, self.gru.bias_hh_l0_reverse], dim=0).contiguous()

        wih_p, whh_p = self.custom_ops_lib.gru_bi_prepack_gatelast_f16_cuda(w_ih, w_hh)
        self._wih_p = wih_p
        self._whh_p = whh_p
        self._bih = b_ih
        self._bhh = b_hh
        self._cache_key = key

    def forward(self, x: torch.Tensor, h0: torch.Tensor):
        can_custom = (
            x.is_cuda and h0.is_cuda and
            x.dtype == torch.float16 and h0.dtype == torch.float16 and
            (not self.batch_first) and
            self.bias and
            self.num_layers == 1 and
            self.gru.bidirectional and
            x.dim() == 3 and h0.dim() == 3 and
            h0.size(0) == 2
        )
        if not can_custom:
            out, _ = self.gru(x, h0)
            return out

        self._maybe_prepack(x)

        y, _hn = self.custom_ops_lib.gru_bi_forward_single_layer_f16_packed_gatelast_cuda(
            x.contiguous(), h0.contiguous(),
            self._wih_p, self._whh_p,
            self._bih, self._bhh
        )
        return y