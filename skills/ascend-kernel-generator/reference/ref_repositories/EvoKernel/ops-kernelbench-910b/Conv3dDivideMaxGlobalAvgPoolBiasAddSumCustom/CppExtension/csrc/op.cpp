
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_divide_max_global_avg_pool_bias_add_sum_custom_impl_npu(
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
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (Cout,Cin,Kd,Kh,Kw)");
    TORCH_CHECK(bias.dim() == 4, "bias must be 4D (Cout,1,1,1)");

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

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");
    TORCH_CHECK(Kd == 3 && Kh == 3 && Kw == 3, "custom op specialized for kernel_size=(3,3,3)");

    TORCH_CHECK(bias.size(0) == Cout && bias.size(1) == 1 && bias.size(2) == 1 && bias.size(3) == 1,
                "bias must be [Cout,1,1,1]");

    TORCH_CHECK(N == 128, "custom op specialized for batch_size=128");
    TORCH_CHECK(Cin == 8 && Cout == 16, "custom op specialized for Cin=8,Cout=16");
    TORCH_CHECK(Din == 16 && Hin == 64 && Win == 64, "custom op specialized for input D/H/W = 16/64/64");

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

    at::Tensor y = at::empty({N, 1, 1, 1}, x.options());
    EXEC_NPU_CMD(aclnnConv3dDivideMaxGlobalAvgPoolBiasAddSumCustom, x, weight, conv_bias, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_divide_max_global_avg_pool_bias_add_sum_custom",
           &conv3d_divide_max_global_avg_pool_bias_add_sum_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_divide_max_global_avg_pool_bias_add_sum_custom",
          &conv3d_divide_max_global_avg_pool_bias_add_sum_custom_impl_npu,
          "conv3d_divide_max_global_avg_pool_bias_add_sum_custom (NPU)");
}
