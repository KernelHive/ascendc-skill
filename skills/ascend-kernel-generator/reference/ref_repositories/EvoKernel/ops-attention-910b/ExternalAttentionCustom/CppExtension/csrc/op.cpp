
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor external_attention_custom_impl_npu(const at::Tensor& x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "external_attention_custom: x must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "external_attention_custom: only float32 supported");
    TORCH_CHECK(x.dim() == 3,
                "external_attention_custom: expected 3D tensor [bs, n, s]");
    TORCH_CHECK(x.is_contiguous(),
                "external_attention_custom: input must be contiguous");

    const auto n  = x.size(1);
    const auto s  = x.size(2);
    TORCH_CHECK(n > 0 && s > 0, "external_attention_custom: invalid shape");
    TORCH_CHECK(n <= 49, "external_attention_custom: n too large for this specialized kernel");
    TORCH_CHECK(s == 64, "external_attention_custom: this optimized kernel requires s==64");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnExternalAttentionCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("external_attention_custom", &external_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("external_attention_custom", &external_attention_custom_impl_npu,
          "ExternalAttention fused normalization (softmax dim=1 then L1 dim=2) on NPU");
}
