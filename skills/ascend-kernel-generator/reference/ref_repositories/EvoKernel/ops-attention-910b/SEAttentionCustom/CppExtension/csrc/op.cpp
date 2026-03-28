
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor se_attention_impl_npu(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& w2) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "se_attention_custom: x must be on NPU");
    TORCH_CHECK(w1.device().type() == c10::DeviceType::PrivateUse1, "se_attention_custom: w1 must be on NPU");
    TORCH_CHECK(w2.device().type() == c10::DeviceType::PrivateUse1, "se_attention_custom: w2 must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "se_attention_custom: only float32 supported");
    TORCH_CHECK(w1.scalar_type() == at::kFloat, "se_attention_custom: only float32 supported");
    TORCH_CHECK(w2.scalar_type() == at::kFloat, "se_attention_custom: only float32 supported");

    TORCH_CHECK(x.dim() == 4, "se_attention_custom: x must be [B,C,H,W]");
    TORCH_CHECK(w1.dim() == 2, "se_attention_custom: w1 must be [R,C]");
    TORCH_CHECK(w2.dim() == 2, "se_attention_custom: w2 must be [C,R]");

    TORCH_CHECK(x.is_contiguous(), "se_attention_custom: x must be contiguous (NCHW)");
    TORCH_CHECK(w1.is_contiguous(), "se_attention_custom: w1 must be contiguous");
    TORCH_CHECK(w2.is_contiguous(), "se_attention_custom: w2 must be contiguous");

    auto B = x.size(0);
    auto C = x.size(1);
    TORCH_CHECK(x.size(2) > 0 && x.size(3) > 0, "se_attention_custom: H/W must be positive");

    TORCH_CHECK(w1.size(1) == C, "se_attention_custom: w1.shape[1] must equal C");
    TORCH_CHECK(w2.size(0) == C, "se_attention_custom: w2.shape[0] must equal C");
    TORCH_CHECK(w2.size(1) == w1.size(0), "se_attention_custom: w2.shape[1] must equal w1.shape[0] (R)");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSEAttentionCustom, x, w1, w2, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("se_attention_custom", &se_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("se_attention_custom", &se_attention_impl_npu, "SE Attention fused op (GAP+FC+ReLU+FC+Sigmoid+Scale) on NPU");
}
