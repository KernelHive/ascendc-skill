
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convt_out_dim(int64_t in, int64_t stride, int64_t pad, int64_t k, int64_t dil, int64_t out_pad) {
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

at::Tensor conv_transpose3d_sum_residual_add_multiply_residual_add_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt,
    const at::Tensor& bias) // [Cout,1,1,1]
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
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (Cin,Cout,Kd,Kh,Kw)");
    TORCH_CHECK(bias.dim() == 4, "bias must be 4D (Cout,1,1,1)");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Din = x.size(2);
    const int64_t Hin = x.size(3);
    const int64_t Win = x.size(4);

    const int64_t wCin = weight.size(0);
    const int64_t Cout = weight.size(1);
    const int64_t Kd   = weight.size(2);
    const int64_t Kh   = weight.size(3);
    const int64_t Kw   = weight.size(4);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");
    TORCH_CHECK(Kd == 3 && Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    TORCH_CHECK(bias.size(0) == Cout && bias.size(1) == 1 && bias.size(2) == 1 && bias.size(3) == 1,
                "bias must be [Cout,1,1,1]");

    constexpr int64_t STR = 2;
    constexpr int64_t PAD = 1;
    constexpr int64_t DIL = 1;
    constexpr int64_t OUT_PAD = 1;

    TORCH_CHECK(N == 16, "custom op specialized for batch_size=16");
    TORCH_CHECK(Cin == 32 && Cout == 64, "custom op specialized for Cin=32,Cout=64");
    TORCH_CHECK(Din == 16 && Hin == 32 && Win == 32, "custom op specialized for input D/H/W = 16/32/32");

    const int64_t Dout = convt_out_dim(Din, STR, PAD, Kd, DIL, OUT_PAD);
    const int64_t Hout = convt_out_dim(Hin, STR, PAD, Kh, DIL, OUT_PAD);
    const int64_t Wout = convt_out_dim(Win, STR, PAD, Kw, DIL, OUT_PAD);
    TORCH_CHECK(Dout == 32 && Hout == 64 && Wout == 64, "unexpected convT output shape for specialized params");

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

    at::Tensor y = at::empty({N, Cout, Dout, Hout, Wout}, x.options());
    EXEC_NPU_CMD(aclnnConvTranspose3dSumResidualAddMultiplyResidualAddCustom, x, weight, conv_bias, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose3d_sum_residual_add_multiply_residual_add_custom",
           &conv_transpose3d_sum_residual_add_multiply_residual_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_sum_residual_add_multiply_residual_add_custom",
          &conv_transpose3d_sum_residual_add_multiply_residual_add_custom_impl_npu,
          "conv_transpose3d_sum_residual_add_multiply_residual_add_custom (NPU)");
}
