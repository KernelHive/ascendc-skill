
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor scaled_dot_product_attention_custom_impl_npu(const at::Tensor& q,
                                                        const at::Tensor& k,
                                                        const at::Tensor& v,
                                                        int64_t d_k) {
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "scaled_dot_product_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "scaled_dot_product_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "scaled_dot_product_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "scaled_dot_product_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "scaled_dot_product_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "scaled_dot_product_attention_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "scaled_dot_product_attention_custom: expected 4D tensors [B,H,S,D]");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "scaled_dot_product_attention_custom: inputs must be contiguous");

    const auto B = q.size(0);
    const auto H = q.size(1);
    const auto S = q.size(2);
    const auto D = q.size(3);

    TORCH_CHECK(d_k > 0, "scaled_dot_product_attention_custom: d_k must be positive");
    TORCH_CHECK(D == d_k, "scaled_dot_product_attention_custom: q.size(3) must equal d_k");

    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(2) == S && k.size(3) == D,
                "scaled_dot_product_attention_custom: k must match q shape [B,H,S,D]");
    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == S,
                "scaled_dot_product_attention_custom: v must match q shape [B,H,S,*]");

    TORCH_CHECK(v.size(3) == D,
                "scaled_dot_product_attention_custom: this fused op requires d_v == d_k (v.size(3) must equal q.size(3))");

    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnScaledDotProductAttentionCustom, q, k, v, d_k, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention_custom", &scaled_dot_product_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_dot_product_attention_custom", &scaled_dot_product_attention_custom_impl_npu,
          "Scaled Dot-Product Attention (fused) on NPU, float32, dropout=0, requires d_v==d_k");
}
