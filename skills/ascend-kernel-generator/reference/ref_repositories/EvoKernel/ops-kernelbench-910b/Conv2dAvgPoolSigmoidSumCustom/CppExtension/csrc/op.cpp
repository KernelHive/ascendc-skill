
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_floor(int64_t in, int64_t k, int64_t s, int64_t p, int64_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

at::Tensor conv2d_avg_pool_sigmoid_sum_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous [Cout,Cin,Kh,Kw]");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cout,Cin,Kh,Kw)");

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");

    // Specialization guardrails (match tiling/kernel/model)
    TORCH_CHECK(N == 128, "custom op specialized for batch_size=128");
    TORCH_CHECK(Cin == 8 && Cout == 64, "custom op specialized for Cin=8, Cout=64");
    TORCH_CHECK(Hin == 384 && Win == 384, "custom op specialized for H=W=384");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    // Fixed conv hyperparams: stride=1,pad=0,dil=1
    constexpr int64_t STR = 1, PAD = 0, DIL = 1;
    const int64_t Hconv = out_floor(Hin, Kh, STR, PAD, DIL);
    const int64_t Wconv = out_floor(Win, Kw, STR, PAD, DIL);
    TORCH_CHECK(Hconv == 382 && Wconv == 382, "unexpected conv output shape for specialized params");

    // Fixed avgpool hyperparams: k=4,s=4,p=0,d=1
    constexpr int64_t PK = 4, PS = 4, PP = 0, PD = 1;
    const int64_t Ho = out_floor(Hconv, PK, PS, PP, PD);
    const int64_t Wo = out_floor(Wconv, PK, PS, PP, PD);
    TORCH_CHECK(Ho == 95 && Wo == 95, "unexpected pool output shape for specialized params");

    // Always pass defined bias to match kernel signature.
    at::Tensor conv_bias;
    if (conv_bias_opt.has_value() && conv_bias_opt.value().defined()) {
        conv_bias = conv_bias_opt.value();
        TORCH_CHECK(conv_bias.device().is_privateuseone(), "conv_bias must be on NPU (PrivateUse1)");
        TORCH_CHECK(conv_bias.scalar_type() == at::kFloat, "conv_bias must be float32");
        TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
        TORCH_CHECK(conv_bias.dim() == 1 && conv_bias.size(0) == Cout, "conv_bias must be 1D [Cout]");
    } else {
        conv_bias = at::zeros({Cout}, x.options());
    }

    at::Tensor y = at::empty({N}, x.options());
    EXEC_NPU_CMD(aclnnConv2dAvgPoolSigmoidSumCustom, x, weight, conv_bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_avg_pool_sigmoid_sum_custom",
           &conv2d_avg_pool_sigmoid_sum_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_avg_pool_sigmoid_sum_custom",
          &conv2d_avg_pool_sigmoid_sum_custom_impl_npu,
          "conv2d_avg_pool_sigmoid_sum_custom (NPU)");
}
