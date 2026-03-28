
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_1d_param(const at::Tensor& t, const char* name, int64_t expected)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.numel() == expected, name, " must have length ", expected);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor conv2d_group_norm_scale_max_pool_clamp_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& gn_gamma,
    const at::Tensor& gn_beta,
    const at::Tensor& scale_1d)
{
    TORCH_CHECK(x.defined(), "x must be defined");
    TORCH_CHECK(weight.defined(), "weight must be defined");
    TORCH_CHECK(bias.defined(), "bias must be defined");
    TORCH_CHECK(gn_gamma.defined(), "gn_gamma must be defined");
    TORCH_CHECK(gn_beta.defined(), "gn_beta must be defined");
    TORCH_CHECK(scale_1d.defined(), "scale must be defined");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");
    TORCH_CHECK(gn_gamma.device().type() == c10::DeviceType::PrivateUse1, "gn_gamma must be on NPU");
    TORCH_CHECK(gn_beta.device().type() == c10::DeviceType::PrivateUse1, "gn_beta must be on NPU");
    TORCH_CHECK(scale_1d.device().type() == c10::DeviceType::PrivateUse1, "scale must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(gn_gamma.scalar_type() == at::kFloat, "gn_gamma must be float32");
    TORCH_CHECK(gn_beta.scalar_type() == at::kFloat, "gn_beta must be float32");
    TORCH_CHECK(scale_1d.scalar_type() == at::kFloat, "scale must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(gn_gamma.is_contiguous(), "gn_gamma must be contiguous");
    TORCH_CHECK(gn_beta.is_contiguous(), "gn_beta must be contiguous");
    TORCH_CHECK(scale_1d.is_contiguous(), "scale must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [Cout,Cin,Kh,Kw]");

    TORCH_CHECK(x.size(1) == 8, "x.size(1) must be 8");
    TORCH_CHECK(x.size(2) == 128 && x.size(3) == 128, "x spatial must be [128,128]");

    TORCH_CHECK(weight.size(0) == 64 && weight.size(1) == 8 &&
                weight.size(2) == 3 && weight.size(3) == 3,
                "weight must be [64,8,3,3]");

    check_1d_param(bias, "bias", 64);
    check_1d_param(gn_gamma, "gn_gamma", 64);
    check_1d_param(gn_beta, "gn_beta", 64);
    check_1d_param(scale_1d, "scale", 64);

    auto y = at::empty({x.size(0), 64, 31, 31}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConv2dGroupNormScaleMaxPoolClampCustom, x, weight, bias, gn_gamma, gn_beta, scale_1d, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_group_norm_scale_max_pool_clamp_custom",
           &conv2d_group_norm_scale_max_pool_clamp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_group_norm_scale_max_pool_clamp_custom",
          &conv2d_group_norm_scale_max_pool_clamp_custom_impl_npu,
          "conv2d_group_norm_scale_max_pool_clamp_custom(x, weight, bias, gn_gamma, gn_beta, scale) -> fused Conv2d+GroupNorm+Scale+MaxPool+Clamp (NPU, specialized)");
}
