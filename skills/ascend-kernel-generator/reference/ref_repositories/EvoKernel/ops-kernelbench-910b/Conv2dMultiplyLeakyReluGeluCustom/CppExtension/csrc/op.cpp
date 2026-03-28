
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_dim_floor(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil) {
    const int64_t eff = dil * (k - 1) + 1;
    TORCH_CHECK(in + 2 * pad >= eff, "Invalid conv params produce negative output dim");
    return (in + 2 * pad - eff) / stride + 1;
}

at::Tensor conv2d_multiply_leaky_relu_gelu_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& multiplier)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(multiplier.device().is_privateuseone(), "multiplier must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(multiplier.scalar_type() == at::kFloat, "multiplier must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(multiplier.is_contiguous() || multiplier.numel() == weight.size(0),
                "multiplier must be contiguous or viewable to [Cout]");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cout,Cin/groups,Kh,Kw)");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kh   = weight.size(2);
    const int64_t Kw   = weight.size(3);

    constexpr int64_t STR = 1;
    constexpr int64_t PAD = 0;
    constexpr int64_t DIL = 1;
    constexpr int64_t GRP = 1;

    TORCH_CHECK(GRP == 1, "custom op specialized for groups=1");
    TORCH_CHECK(wCin == Cin, "weight Cin mismatch for groups=1");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    TORCH_CHECK(N == 64, "custom op specialized for batch_size=64");
    TORCH_CHECK(Cin == 64 && Cout == 64, "custom op specialized for Cin=64,Cout=64");
    TORCH_CHECK(Hin == 256 && Win == 256, "custom op specialized for input H/W = 256/256");

    TORCH_CHECK(multiplier.numel() == Cout, "multiplier must have exactly Cout elements");
    TORCH_CHECK(multiplier.dim() == 3 || multiplier.dim() == 1,
                "multiplier must be [Cout,1,1] or [Cout]");

    const int64_t Hout = out_dim_floor(Hin, Kh, PAD, STR, DIL);
    const int64_t Wout = out_dim_floor(Win, Kw, PAD, STR, DIL);
    TORCH_CHECK(Hout == 254 && Wout == 254, "unexpected output shape for specialized params");

    at::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == Cout, "bias must be 1D [Cout]");
    } else {
        bias = at::zeros({Cout}, x.options());
    }

    at::Tensor mult_flat = multiplier.view({Cout}).contiguous();
    TORCH_CHECK(mult_flat.is_contiguous(), "mult_flat must be contiguous");

    at::Tensor y = at::empty({N, Cout, Hout, Wout}, x.options());
    EXEC_NPU_CMD(aclnnConv2dMultiplyLeakyReluGeluCustom, x, weight, bias, mult_flat, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_multiply_leaky_relu_gelu_custom",
           &conv2d_multiply_leaky_relu_gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_multiply_leaky_relu_gelu_custom",
          &conv2d_multiply_leaky_relu_gelu_custom_impl_npu,
          "conv2d_multiply_leaky_relu_gelu_custom (NPU)");
}
