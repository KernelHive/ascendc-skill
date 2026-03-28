
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

at::Tensor conv3d_group_norm_mean_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& gamma,
    const at::Tensor& beta)
{
    TORCH_CHECK(x.defined(), "x must be defined");
    TORCH_CHECK(weight.defined(), "weight must be defined");
    TORCH_CHECK(bias.defined(), "bias must be defined");
    TORCH_CHECK(gamma.defined(), "gamma must be defined");
    TORCH_CHECK(beta.defined(), "beta must be defined");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");
    TORCH_CHECK(gamma.device().type() == c10::DeviceType::PrivateUse1, "gamma must be on NPU");
    TORCH_CHECK(beta.device().type() == c10::DeviceType::PrivateUse1, "beta must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cout,Cin,K,K,K]");

    // Specialized contract (matches host tiling/kernel)
    TORCH_CHECK(x.size(1) == 3, "x.size(1) must be 3");
    TORCH_CHECK(x.size(2) == 24 && x.size(3) == 32 && x.size(4) == 32,
                "x spatial must be [24,32,32]");

    TORCH_CHECK(weight.size(0) == 24 && weight.size(1) == 3 &&
                weight.size(2) == 3 && weight.size(3) == 3 && weight.size(4) == 3,
                "weight must be [24,3,3,3,3]");

    check_1d_param(bias, "bias", 24);
    check_1d_param(gamma, "gamma", 24);
    check_1d_param(beta, "beta", 24);

    auto y = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConv3dGroupNormMeanCustom, x, weight, bias, gamma, beta, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_group_norm_mean_custom", &conv3d_group_norm_mean_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_group_norm_mean_custom",
          &conv3d_group_norm_mean_custom_impl_npu,
          "conv3d_group_norm_mean_custom(x, weight, bias, gamma, beta) -> fused Conv3d+GroupNorm+Mean (NPU, specialized)");
}
