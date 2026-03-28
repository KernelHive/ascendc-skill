
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gated_channel_transform_impl_npu(const at::Tensor& x,
                                           const at::Tensor& alpha,
                                           const at::Tensor& gamma,
                                           const at::Tensor& beta,
                                           double epsilon)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "gated_channel_transform_custom: x must be on NPU");
    TORCH_CHECK(alpha.device().type() == c10::DeviceType::PrivateUse1, "gated_channel_transform_custom: alpha must be on NPU");
    TORCH_CHECK(gamma.device().type() == c10::DeviceType::PrivateUse1, "gated_channel_transform_custom: gamma must be on NPU");
    TORCH_CHECK(beta.device().type() == c10::DeviceType::PrivateUse1, "gated_channel_transform_custom: beta must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "gated_channel_transform_custom: x must be float32");
    TORCH_CHECK(alpha.scalar_type() == at::kFloat, "gated_channel_transform_custom: alpha must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gated_channel_transform_custom: gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "gated_channel_transform_custom: beta must be float32");

    TORCH_CHECK(x.dim() == 4, "gated_channel_transform_custom: x must be [N,C,H,W]");
    TORCH_CHECK(x.is_contiguous(), "gated_channel_transform_custom: x must be contiguous NCHW");

    const auto C = x.size(1);
    TORCH_CHECK(C > 0, "gated_channel_transform_custom: C must be > 0");
    TORCH_CHECK(x.size(2) > 0 && x.size(3) > 0, "gated_channel_transform_custom: H/W must be > 0");

    // We pass params as flat [C] (converted from [1,C,1,1] in ModelNew)
    TORCH_CHECK(alpha.dim() == 1 && gamma.dim() == 1 && beta.dim() == 1,
                "gated_channel_transform_custom: alpha/gamma/beta must be 1D [C]");
    TORCH_CHECK(alpha.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "gated_channel_transform_custom: alpha/gamma/beta must be contiguous");
    TORCH_CHECK(alpha.size(0) == C && gamma.size(0) == C && beta.size(0) == C,
                "gated_channel_transform_custom: alpha/gamma/beta length must equal C");

    TORCH_CHECK(epsilon >= 0.0, "gated_channel_transform_custom: epsilon must be >= 0");

    at::Tensor eps_t = at::empty({}, x.options().dtype(at::kFloat));
    eps_t.fill_(static_cast<float>(epsilon));

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnGatedChannelTransformCustom, x, alpha, gamma, beta, eps_t, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("gated_channel_transform_custom(Tensor x, Tensor alpha, Tensor gamma, Tensor beta, float epsilon) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gated_channel_transform_custom", &gated_channel_transform_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gated_channel_transform_custom", &gated_channel_transform_impl_npu,
          "Gated Channel Transform fused op (mode='l2') on NPU");
}
