
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_1d_param(const at::Tensor& t, const char* name, int64_t expected)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.numel() == expected, name, " must have length ", expected);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor conv3d_group_norm_min_clamp_dropout_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& gn_gamma,
    const at::Tensor& gn_beta)
{
    TORCH_CHECK(x.defined(), "x must be defined");
    TORCH_CHECK(weight.defined(), "weight must be defined");
    TORCH_CHECK(bias.defined(), "bias must be defined");
    TORCH_CHECK(gn_gamma.defined(), "gn_gamma must be defined");
    TORCH_CHECK(gn_beta.defined(), "gn_beta must be defined");

    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");
    TORCH_CHECK(gn_gamma.device().is_privateuseone(), "gn_gamma must be on NPU (PrivateUse1)");
    TORCH_CHECK(gn_beta.device().is_privateuseone(), "gn_beta must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(gn_gamma.scalar_type() == at::kFloat, "gn_gamma must be float32");
    TORCH_CHECK(gn_beta.scalar_type() == at::kFloat, "gn_beta must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(gn_gamma.is_contiguous(), "gn_gamma must be contiguous");
    TORCH_CHECK(gn_beta.is_contiguous(), "gn_beta must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cout,Cin,K,K,K]");

    // Specialized benchmark contract
    TORCH_CHECK(x.size(1) == 3, "x.size(1) must be 3");
    TORCH_CHECK(x.size(2) == 16 && x.size(3) == 64 && x.size(4) == 64,
                "x must be [N,3,16,64,64]");
    TORCH_CHECK(weight.size(0) == 16 && weight.size(1) == 3 &&
                weight.size(2) == 3 && weight.size(3) == 3 && weight.size(4) == 3,
                "weight must be [16,3,3,3,3]");

    check_1d_param(bias, "bias", 16);
    check_1d_param(gn_gamma, "gn_gamma", 16);
    check_1d_param(gn_beta, "gn_beta", 16);

    auto y = at::empty({x.size(0), 16, 14, 62, 62}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConv3dGroupNormMinClampDropoutCustom, x, weight, bias, gn_gamma, gn_beta, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_group_norm_min_clamp_dropout_custom",
           &conv3d_group_norm_min_clamp_dropout_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_group_norm_min_clamp_dropout_custom",
          &conv3d_group_norm_min_clamp_dropout_custom_impl_npu,
          "conv3d_group_norm_min_clamp_dropout_custom (NPU)");
}
