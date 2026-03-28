
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor residual_attention_impl_npu(const at::Tensor& x, const at::Tensor& la)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "residual_attention_custom: x must be on NPU");
    TORCH_CHECK(la.device().type() == c10::DeviceType::PrivateUse1, "residual_attention_custom: la must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "residual_attention_custom: only float32 supported");
    TORCH_CHECK(la.scalar_type() == at::kFloat, "residual_attention_custom: la must be float32");

    TORCH_CHECK(x.dim() == 4, "residual_attention_custom: x must be [B,C,H,W]");
    TORCH_CHECK(x.is_contiguous(), "residual_attention_custom: x must be contiguous (NCHW)");

    TORCH_CHECK(la.numel() == 1, "residual_attention_custom: la must have numel()==1");
    TORCH_CHECK(la.is_contiguous(), "residual_attention_custom: la must be contiguous");

    const auto B = x.size(0);
    const auto C = x.size(1);
    TORCH_CHECK(x.size(2) > 0 && x.size(3) > 0, "residual_attention_custom: H/W must be positive");

    at::Tensor y = at::empty({B, C}, x.options());
    EXEC_NPU_CMD(aclnnResidualAttentionCustom, x, la, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("residual_attention_custom", &residual_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("residual_attention_custom", &residual_attention_impl_npu,
          "Residual Attention fused reduce op: mean(x,HW) + la*max(x,HW) over spatial dims (NPU)");
}
