
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor srm_impl_npu(const at::Tensor& x,
                        const at::Tensor& cfc_weight,
                        const at::Tensor& bn_weight,
                        const at::Tensor& bn_bias,
                        const at::Tensor& bn_running_mean,
                        const at::Tensor& bn_running_var,
                        double eps)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "srm_custom: x must be on NPU");
    TORCH_CHECK(cfc_weight.device().type() == c10::DeviceType::PrivateUse1, "srm_custom: cfc_weight must be on NPU");
    TORCH_CHECK(bn_weight.device().type() == c10::DeviceType::PrivateUse1, "srm_custom: bn_weight must be on NPU");
    TORCH_CHECK(bn_bias.device().type() == c10::DeviceType::PrivateUse1, "srm_custom: bn_bias must be on NPU");
    TORCH_CHECK(bn_running_mean.device().type() == c10::DeviceType::PrivateUse1, "srm_custom: bn_running_mean must be on NPU");
    TORCH_CHECK(bn_running_var.device().type() == c10::DeviceType::PrivateUse1, "srm_custom: bn_running_var must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "srm_custom: only float32 supported for x");
    TORCH_CHECK(cfc_weight.scalar_type() == at::kFloat, "srm_custom: only float32 supported for cfc_weight");
    TORCH_CHECK(bn_weight.scalar_type() == at::kFloat, "srm_custom: only float32 supported for bn_weight");
    TORCH_CHECK(bn_bias.scalar_type() == at::kFloat, "srm_custom: only float32 supported for bn_bias");
    TORCH_CHECK(bn_running_mean.scalar_type() == at::kFloat, "srm_custom: only float32 supported for bn_running_mean");
    TORCH_CHECK(bn_running_var.scalar_type() == at::kFloat, "srm_custom: only float32 supported for bn_running_var");

    TORCH_CHECK(x.dim() == 4, "srm_custom: x must be [B,C,H,W]");
    TORCH_CHECK(cfc_weight.dim() == 3, "srm_custom: cfc_weight must be [C,1,2]");
    TORCH_CHECK(bn_weight.dim() == 1, "srm_custom: bn_weight must be [C]");
    TORCH_CHECK(bn_bias.dim() == 1, "srm_custom: bn_bias must be [C]");
    TORCH_CHECK(bn_running_mean.dim() == 1, "srm_custom: bn_running_mean must be [C]");
    TORCH_CHECK(bn_running_var.dim() == 1, "srm_custom: bn_running_var must be [C]");

    TORCH_CHECK(x.is_contiguous(), "srm_custom: x must be contiguous NCHW");
    TORCH_CHECK(cfc_weight.is_contiguous(), "srm_custom: cfc_weight must be contiguous");
    TORCH_CHECK(bn_weight.is_contiguous(), "srm_custom: bn_weight must be contiguous");
    TORCH_CHECK(bn_bias.is_contiguous(), "srm_custom: bn_bias must be contiguous");
    TORCH_CHECK(bn_running_mean.is_contiguous(), "srm_custom: bn_running_mean must be contiguous");
    TORCH_CHECK(bn_running_var.is_contiguous(), "srm_custom: bn_running_var must be contiguous");

    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    TORCH_CHECK(C == 512, "srm_custom: specialized for C=512");
    TORCH_CHECK(H == 7 && W == 7, "srm_custom: specialized for H=W=7");
    TORCH_CHECK(cfc_weight.size(0) == C && cfc_weight.size(1) == 1 && cfc_weight.size(2) == 2,
                "srm_custom: cfc_weight must be [C,1,2]");
    TORCH_CHECK(bn_weight.size(0) == C, "srm_custom: bn_weight must be [C]");
    TORCH_CHECK(bn_bias.size(0) == C, "srm_custom: bn_bias must be [C]");
    TORCH_CHECK(bn_running_mean.size(0) == C, "srm_custom: bn_running_mean must be [C]");
    TORCH_CHECK(bn_running_var.size(0) == C, "srm_custom: bn_running_var must be [C]");
    TORCH_CHECK(eps > 0.0, "srm_custom: eps must be > 0");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSRMCustom, x, cfc_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, y, eps);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("srm_custom(Tensor x, Tensor cfc_weight, Tensor bn_weight, Tensor bn_bias, Tensor bn_running_mean, Tensor bn_running_var, float eps=1e-5) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("srm_custom", &srm_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("srm_custom", &srm_impl_npu,
          "SRM fused op (style pooling + depthwise conv1d k=2 + BN inference + sigmoid + scale) specialized for C=512,H=W=7 on NPU",
          py::arg("x"),
          py::arg("cfc_weight"),
          py::arg("bn_weight"),
          py::arg("bn_bias"),
          py::arg("bn_running_mean"),
          py::arg("bn_running_var"),
          py::arg("eps") = 1e-5);
}
