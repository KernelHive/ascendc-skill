
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor spatial_group_enhance_impl_npu(const at::Tensor& x,
                                         const at::Tensor& weight,
                                         const at::Tensor& bias) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "spatial_group_enhance_custom: x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "spatial_group_enhance_custom: weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "spatial_group_enhance_custom: bias must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "spatial_group_enhance_custom: only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "spatial_group_enhance_custom: only float32 supported");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "spatial_group_enhance_custom: only float32 supported");

    TORCH_CHECK(x.dim() == 4, "spatial_group_enhance_custom: x must be [B,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "spatial_group_enhance_custom: weight must be [1,G,1,1]");
    TORCH_CHECK(bias.dim() == 4, "spatial_group_enhance_custom: bias must be [1,G,1,1]");

    TORCH_CHECK(x.is_contiguous(), "spatial_group_enhance_custom: x must be contiguous (NCHW)");
    TORCH_CHECK(weight.is_contiguous(), "spatial_group_enhance_custom: weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "spatial_group_enhance_custom: bias must be contiguous");

    TORCH_CHECK(weight.size(0) == 1 && weight.size(2) == 1 && weight.size(3) == 1,
                "spatial_group_enhance_custom: weight must have shape [1,G,1,1]");
    TORCH_CHECK(bias.size(0) == 1 && bias.size(2) == 1 && bias.size(3) == 1,
                "spatial_group_enhance_custom: bias must have shape [1,G,1,1]");
    TORCH_CHECK(bias.size(1) == weight.size(1), "spatial_group_enhance_custom: bias G must equal weight G");

    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0, "spatial_group_enhance_custom: invalid x shape");

    auto G = weight.size(1);
    TORCH_CHECK(G > 0, "spatial_group_enhance_custom: G must be > 0");
    TORCH_CHECK((C % G) == 0, "spatial_group_enhance_custom: C must be divisible by G");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSpatialGroupEnhanceCustom, x, weight, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("spatial_group_enhance_custom", &spatial_group_enhance_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_group_enhance_custom", &spatial_group_enhance_impl_npu,
          "Spatial Group Enhance fused op (SGE) on NPU");
}
