
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_dim_floor(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil) {
    const int64_t eff = dil * (k - 1) + 1;
    TORCH_CHECK(in + 2 * pad >= eff, "Invalid conv params produce negative output dim");
    return (in + 2 * pad - eff) / stride + 1;
}

at::Tensor conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt,
    const at::Tensor& bias)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (Cout,Cin/groups,Kd,Kh,Kw)");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Din = x.size(2);
    const int64_t Hin = x.size(3);
    const int64_t Win = x.size(4);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kd   = weight.size(2);
    const int64_t Kh   = weight.size(3);
    const int64_t Kw   = weight.size(4);

    constexpr int64_t STR = 1;
    constexpr int64_t PAD = 0;
    constexpr int64_t DIL = 1;
    constexpr int64_t GRP = 1;

    TORCH_CHECK(GRP == 1, "custom op specialized for groups=1");
    TORCH_CHECK(wCin == Cin, "weight Cin mismatch for groups=1");
    TORCH_CHECK(Kd == 3 && Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    TORCH_CHECK(N == 64, "custom op specialized for batch_size=64");
    TORCH_CHECK(Cin == 8 && Cout == 32, "custom op specialized for Cin=8,Cout=32");
    TORCH_CHECK(Din == 32 && Hin == 64 && Win == 64, "custom op specialized for input D/H/W = 32/64/64");

    TORCH_CHECK(bias.numel() == Cout, "bias must have Cout elements (e.g. [Cout,1,1,1])");
    TORCH_CHECK(bias.dim() == 4 || bias.dim() == 1, "bias must be [Cout,1,1,1] or [Cout]");

    const int64_t Dout = out_dim_floor(Din, Kd, PAD, STR, DIL);
    const int64_t Hout = out_dim_floor(Hin, Kh, PAD, STR, DIL);
    const int64_t Wout = out_dim_floor(Win, Kw, PAD, STR, DIL);
    TORCH_CHECK(Dout == 30 && Hout == 62 && Wout == 62, "unexpected output shape for specialized params");

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

    at::Tensor bias_flat = bias.view({Cout}).contiguous();

    at::Tensor y = at::empty({N, Cout, Dout, Hout, Wout}, x.options());
    EXEC_NPU_CMD(aclnnConv3dReluLeakyReluGeluSigmoidBiasAddCustom, x, weight, conv_bias, bias_flat, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom",
           &conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom",
          &conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom_impl_npu,
          "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom (NPU)");
}
