import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension
# -------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

#if __CUDA_ARCH__ >= 350
#define LDG(p) __ldg(p)
#else
#define LDG(p) (*(p))
#endif

// -------------------------
// Direct conv kernels (baseline, simple reference).
// -------------------------

__global__ void conv2d_fwd_nchw_k3(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin,3,3]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride, int pad
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int in_y0 = oh * stride - pad;
    int in_x0 = ow * stride - pad;

    float acc = 0.0f;
    int w_oc_base = oc * Cin * 9;

    for (int ic = 0; ic < Cin; ++ic) {
        int w_ic_base = w_oc_base + ic * 9;
        int x_ic_base = ((n * Cin + ic) * Hin) * Win;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int iy = in_y0 + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            int x_row = x_ic_base + iy * Win;
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int ix = in_x0 + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;
                float xv = x[x_row + ix];
                float wv = LDG(w + w_ic_base + ky * 3 + kx);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

__global__ void conv2d_fwd_nchw_k1(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin,1,1] logically flattened
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int iy = oh * stride;
    int ix = ow * stride;

    float acc = 0.0f;
    int w_oc_base = oc * Cin;

    int x_base = ((n * Cin) * Hin + iy) * Win + ix;
    int hw = Hin * Win;
    for (int ic = 0; ic < Cin; ++ic) {
        float xv = x[x_base + ic * hw];
        float wv = LDG(w + w_oc_base + ic);
        acc = fmaf(xv, wv, acc);
    }

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

// -------------------------
// Vectorized elementwise kernels
// -------------------------

static __device__ __forceinline__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// BN-only inplace: x = x*alpha[c] + beta[c]
__global__ __launch_bounds__(256, 4) void bn_inplace_vec(
    float* __restrict__ x,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    int64_t total, int HxW, int W
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    // float4 path only if pointers aligned and W % 4 == 0 (so vectors don't cross rows)
    bool vec4_ok = (W % 4 == 0) && is_aligned_16(x) && is_aligned_16(alpha) && is_aligned_16(beta);

    if (vec4_ok) {
        int64_t total4 = total / 4;
        for (int64_t i4 = tid; i4 < total4; i4 += (int64_t)blockDim.x * gridDim.x) {
            int64_t base = i4 * 4;  // scalar offset
            int c = (int)((base / HxW) % (int64_t)0x7fffffff); // placeholder; corrected below
            // Need true C, can't infer here; we instead compute c via division by HxW and mod C,
            // but C isn't passed. So pass HxW only isn't enough.
            // This kernel is not used; kept for completeness.
        }
        return;
    }

    for (int64_t i = tid; i < total; i += (int64_t)blockDim.x * gridDim.x) {
        // Derive c from i: i = (((n*C + c)*H + h)*W + w)
        // Let HxW = H*W, then c = (i / HxW) % C. But C not passed; this kernel isn't used.
        // Kept only to avoid compilation warnings.
        x[i] = x[i];
    }
}

// BN-only out-of-place, vectorized over W (requires C to compute channel)
__global__ __launch_bounds__(256, 4) void bn_out_vec(
    const float* __restrict__ in,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int64_t total, int C, int HxW, int W
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    bool vec4_ok = (W % 4 == 0) && is_aligned_16(in) && is_aligned_16(out);

    if (vec4_ok) {
        int64_t total4 = total / 4;
        const float4* in4 = reinterpret_cast<const float4*>(in);
        float4* out4 = reinterpret_cast<float4*>(out);

        for (int64_t i4 = tid; i4 < total4; i4 += (int64_t)blockDim.x * gridDim.x) {
            int64_t base = i4 * 4;              // scalar offset
            int c = (int)((base / HxW) % C);

            float a = LDG(alpha + c);
            float b = LDG(beta + c);

            float4 v = in4[i4];
            v.x = fmaf(v.x, a, b);
            v.y = fmaf(v.y, a, b);
            v.z = fmaf(v.z, a, b);
            v.w = fmaf(v.w, a, b);
            out4[i4] = v;
        }
        return;
    }

    for (int64_t i = tid; i < total; i += (int64_t)blockDim.x * gridDim.x) {
        int c = (int)((i / HxW) % C);
        float a = LDG(alpha + c);
        float b = LDG(beta + c);
        out[i] = fmaf(in[i], a, b);
    }
}

// BN + Add + ReLU: y = relu(x*alpha[c] + beta[c] + res)
__global__ __launch_bounds__(256, 4) void bn_add_relu_vec(
    const float* __restrict__ x,
    const float* __restrict__ res,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int64_t total, int C, int HxW, int W
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    bool vec4_ok = (W % 4 == 0) && is_aligned_16(x) && is_aligned_16(res) && is_aligned_16(y);

    if (vec4_ok) {
        int64_t total4 = total / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* r4 = reinterpret_cast<const float4*>(res);
        float4* y4 = reinterpret_cast<float4*>(y);

        for (int64_t i4 = tid; i4 < total4; i4 += (int64_t)blockDim.x * gridDim.x) {
            int64_t base = i4 * 4;
            int c = (int)((base / HxW) % C);
            float a = LDG(alpha + c);
            float b = LDG(beta + c);

            float4 xv = x4[i4];
            float4 rv = r4[i4];

            float4 out;
            float t0 = fmaf(xv.x, a, b) + rv.x; out.x = t0 > 0.f ? t0 : 0.f;
            float t1 = fmaf(xv.y, a, b) + rv.y; out.y = t1 > 0.f ? t1 : 0.f;
            float t2 = fmaf(xv.z, a, b) + rv.z; out.z = t2 > 0.f ? t2 : 0.f;
            float t3 = fmaf(xv.w, a, b) + rv.w; out.w = t3 > 0.f ? t3 : 0.f;

            y4[i4] = out;
        }
        return;
    }

    for (int64_t i = tid; i < total; i += (int64_t)blockDim.x * gridDim.x) {
        int c = (int)((i / HxW) % C);
        float v = fmaf(x[i], LDG(alpha + c), LDG(beta + c)) + res[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

// -------------------------
// Launch helpers
// -------------------------

static inline int get_grid_1d(int64_t n, int block) {
    // Cap grid to avoid excessive launch overhead; rely on grid-stride loop.
    int64_t g = (n + block - 1) / block;
    if (g > 65535) g = 65535;
    if (g < 1) g = 1;
    return (int)g;
}

static inline void launch_conv3(
    torch::Tensor x, torch::Tensor w, torch::Tensor y,
    int stride, int pad, cudaStream_t stream
) {
    int N = (int)x.size(0), Cin=(int)x.size(1), Hin=(int)x.size(2), Win=(int)x.size(3);
    int Cout=(int)w.size(0);
    int Hout=(int)y.size(2), Wout=(int)y.size(3);
    int total = N*Cout*Hout*Wout;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_fwd_nchw_k3<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout, stride, pad
    );
}

static inline void launch_conv1(
    torch::Tensor x, torch::Tensor w, torch::Tensor y,
    int stride, cudaStream_t stream
) {
    int N = (int)x.size(0), Cin=(int)x.size(1), Hin=(int)x.size(2), Win=(int)x.size(3);
    int Cout=(int)w.size(0);
    int Hout=(int)y.size(2), Wout=(int)y.size(3);
    int total = N*Cout*Hout*Wout;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_fwd_nchw_k1<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout, stride
    );
}

// Downsample BN-only (out-of-place) to avoid needing an extra temp for inplace and keep vectorization safe.
static inline void launch_bn_out(
    torch::Tensor in, torch::Tensor alpha, torch::Tensor beta, torch::Tensor out,
    int C, int H, int W, cudaStream_t stream
) {
    int64_t total = in.numel();
    int HxW = H * W;
    int block = 256;
    int grid = get_grid_1d(total, block);
    bn_out_vec<<<grid, block, 0, stream>>>(
        in.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(), out.data_ptr<float>(),
        total, C, HxW, W
    );
}

static inline void launch_bn_add_relu(
    torch::Tensor x, torch::Tensor res, torch::Tensor alpha, torch::Tensor beta, torch::Tensor y,
    int C, int H, int W, cudaStream_t stream
) {
    int64_t total = x.numel();
    int HxW = H * W;
    int block = 256;
    int grid = get_grid_1d(total, block);
    bn_add_relu_vec<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), res.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(),
        total, C, HxW, W
    );
}

torch::Tensor basicblock_forward_cuda(
    torch::Tensor x,            // [N,Cin,H,W]
    torch::Tensor w1,           // [Cout,Cin,3,3]
    torch::Tensor a1,           // [Cout]
    torch::Tensor b1,           // [Cout]
    torch::Tensor w2,           // [Cout,Cout,3,3]
    torch::Tensor a2,           // [Cout]
    torch::Tensor b2,           // [Cout]
    torch::Tensor w_ds,         // [] or [Cout,Cin,1,1]
    torch::Tensor a_ds,         // [] or [Cout]
    torch::Tensor b_ds,         // [] or [Cout]
    int64_t stride1,            // 1 or 2
    bool has_downsample
) {
    CHECK_CUDA(x); CHECK_CUDA(w1); CHECK_CUDA(w2);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w1); CHECK_CONTIGUOUS(w2);
    CHECK_FLOAT(x); CHECK_FLOAT(w1); CHECK_FLOAT(w2);
    CHECK_CUDA(a1); CHECK_CUDA(b1); CHECK_CUDA(a2); CHECK_CUDA(b2);
    CHECK_CONTIGUOUS(a1); CHECK_CONTIGUOUS(b1); CHECK_CONTIGUOUS(a2); CHECK_CONTIGUOUS(b2);
    CHECK_FLOAT(a1); CHECK_FLOAT(b1); CHECK_FLOAT(a2); CHECK_FLOAT(b2);

    TORCH_CHECK(x.dim()==4, "x must be NCHW");
    TORCH_CHECK(w1.dim()==4 && w1.size(2)==3 && w1.size(3)==3, "w1 must be 3x3");
    TORCH_CHECK(w2.dim()==4 && w2.size(2)==3 && w2.size(3)==3, "w2 must be 3x3");

    int64_t N = x.size(0), Cin=x.size(1), Hin=x.size(2), Win=x.size(3);
    int64_t Cout = w1.size(0);
    TORCH_CHECK(w1.size(1)==Cin, "w1 Cin mismatch");
    TORCH_CHECK(w2.size(0)==Cout && w2.size(1)==Cout, "w2 shape mismatch");
    TORCH_CHECK(a1.numel()==Cout && b1.numel()==Cout, "bn1 params mismatch");
    TORCH_CHECK(a2.numel()==Cout && b2.numel()==Cout, "bn2 params mismatch");
    TORCH_CHECK(stride1==1 || stride1==2, "stride1 must be 1 or 2");

    int64_t H1 = (Hin + 2 - 3) / stride1 + 1;
    int64_t W1 = (Win + 2 - 3) / stride1 + 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // conv1
    auto o1 = torch::empty({N, Cout, H1, W1}, x.options());
    launch_conv3(x, w1, o1, (int)stride1, 1, stream);

    // BN+ReLU inplace on o1 via bn_add_relu_vec with res=0? Avoid extra tensor.
    // Keep original scalar kernel for BN+ReLU? Instead: reuse bn_add_relu_vec by pointing res to o1 and subtracting isn't correct.
    // So keep a small scalar BN+ReLU kernel but optimize final epilogue heavily.
    // For simplicity and to keep changes incremental, we keep the existing scalar bn_relu_inplace kernel here.
    // (Defined inline below.)
    {
        // scalar BN+ReLU
        int64_t total = o1.numel();
        int block = 256;
        int grid = get_grid_1d(total, block);
        // Kernel defined as lambda-like static below.
        extern __global__ void bn_relu_inplace_scalar(float*, const float*, const float*, int64_t, int, int);
        bn_relu_inplace_scalar<<<grid, block, 0, stream>>>(
            o1.data_ptr<float>(), a1.data_ptr<float>(), b1.data_ptr<float>(),
            total, (int)(H1*W1), (int)Cout
        );
    }

    // conv2
    auto o2 = torch::empty({N, Cout, H1, W1}, x.options());
    launch_conv3(o1, w2, o2, 1, 1, stream);

    torch::Tensor res = x;
    if (has_downsample) {
        CHECK_CUDA(w_ds); CHECK_CUDA(a_ds); CHECK_CUDA(b_ds);
        CHECK_CONTIGUOUS(w_ds); CHECK_CONTIGUOUS(a_ds); CHECK_CONTIGUOUS(b_ds);
        CHECK_FLOAT(w_ds); CHECK_FLOAT(a_ds); CHECK_FLOAT(b_ds);
        TORCH_CHECK(w_ds.dim()==4 && w_ds.size(2)==1 && w_ds.size(3)==1, "w_ds must be 1x1");
        TORCH_CHECK(w_ds.size(0)==Cout && w_ds.size(1)==Cin, "w_ds shape mismatch");
        TORCH_CHECK(a_ds.numel()==Cout && b_ds.numel()==Cout, "bn_ds params mismatch");

        auto ds_conv = torch::empty({N, Cout, H1, W1}, x.options());
        // conv1 kernel expects weights flattened; use data_ptr directly (no view/contiguous!)
        launch_conv1(x, w_ds, ds_conv, (int)stride1, stream);

        auto ds = torch::empty({N, Cout, H1, W1}, x.options());
        launch_bn_out(ds_conv, a_ds, b_ds, ds, (int)Cout, (int)H1, (int)W1, stream);
        res = ds;
    } else {
        TORCH_CHECK(Cin == Cout && Hin == H1 && Win == W1, "identity shape mismatch without downsample");
    }

    auto y = torch::empty({N, Cout, H1, W1}, x.options());
    launch_bn_add_relu(o2, res, a2, b2, y, (int)Cout, (int)H1, (int)W1, stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Scalar BN+ReLU inplace used for conv1 output (kept simple; conv dominates anyway)
__global__ __launch_bounds__(256, 4) void bn_relu_inplace_scalar(
    float* __restrict__ x,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    int64_t total, int HxW, int C
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = tid; i < total; i += (int64_t)blockDim.x * gridDim.x) {
        int c = (int)((i / HxW) % C);
        float v = fmaf(x[i], LDG(alpha + c), LDG(beta + c));
        x[i] = v > 0.0f ? v : 0.0f;
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor basicblock_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor a2, torch::Tensor b2,
    torch::Tensor w_ds, torch::Tensor a_ds, torch::Tensor b_ds,
    int64_t stride1, bool has_downsample
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_resnet18_basicblock_optvec",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["basicblock_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model definition (uses fused basicblock op)
# -------------------------

class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    @staticmethod
    def _bn_alpha_beta(bn: nn.BatchNorm2d, device, dtype):
        gamma = bn.weight.to(device=device, dtype=dtype)
        beta = bn.bias.to(device=device, dtype=dtype)
        mean = bn.running_mean.to(device=device, dtype=dtype)
        var = bn.running_var.to(device=device, dtype=dtype)
        invstd = torch.rsqrt(var + bn.eps)
        alpha = gamma * invstd
        beta2 = beta - mean * alpha
        return alpha.contiguous(), beta2.contiguous()

    def forward(self, x):
        if (
            x.is_cuda and x.dtype == torch.float32 and (not self.training) and
            self.conv1.weight.is_cuda and self.conv2.weight.is_cuda
        ):
            x = x.contiguous()
            w1 = self.conv1.weight.contiguous()
            w2 = self.conv2.weight.contiguous()
            a1, b1 = self._bn_alpha_beta(self.bn1, x.device, x.dtype)
            a2, b2 = self._bn_alpha_beta(self.bn2, x.device, x.dtype)

            has_ds = self.downsample is not None
            if has_ds:
                ds_conv = self.downsample[0]
                ds_bn = self.downsample[1]
                wds = ds_conv.weight.contiguous()
                ads, bds = self._bn_alpha_beta(ds_bn, x.device, x.dtype)
            else:
                wds = torch.empty(0, device=x.device, dtype=x.dtype)
                ads = torch.empty(0, device=x.device, dtype=x.dtype)
                bds = torch.empty(0, device=x.device, dtype=x.dtype)

            return custom_ops_lib.basicblock_forward_cuda(
                x, w1, a1, b1, w2, a2, b2, wds, ads, bds, int(self.stride), bool(has_ds)
            )

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockNew, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x