
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor multi_head_attention_custom_impl_npu(const at::Tensor& q,
                                                const at::Tensor& k,
                                                const at::Tensor& v) {
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "multi_head_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "multi_head_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "multi_head_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "multi_head_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "multi_head_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "multi_head_attention_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "multi_head_attention_custom: expected 4D tensors [B,H,S,D]");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "multi_head_attention_custom: inputs must be contiguous");

    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "multi_head_attention_custom: q/k/v must have identical shapes [B,H,S,D]");

    TORCH_CHECK(q.size(3) <= 128, "multi_head_attention_custom: D>128 not supported by this kernel");
    TORCH_CHECK(q.size(2) <= 512, "multi_head_attention_custom: S>512 not supported by this kernel");

    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnMultiHeadAttentionCustom, q, k, v, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("multi_head_attention_custom", &multi_head_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_attention_custom", &multi_head_attention_custom_impl_npu,
          "Fused scaled dot-product attention core (Q,K,V)->O on NPU, float32, dropout=0");
}
