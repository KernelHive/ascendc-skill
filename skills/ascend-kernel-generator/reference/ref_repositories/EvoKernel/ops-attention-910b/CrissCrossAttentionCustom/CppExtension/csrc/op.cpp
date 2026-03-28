
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor criss_cross_attention_impl_npu(const at::Tensor& x,
                                         const at::Tensor& out_h,
                                         const at::Tensor& out_w,
                                         const at::Tensor& gamma) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "criss_cross_attention_custom: only float32 supported");
    TORCH_CHECK(out_h.scalar_type() == at::kFloat, "criss_cross_attention_custom: only float32 supported");
    TORCH_CHECK(out_w.scalar_type() == at::kFloat, "criss_cross_attention_custom: only float32 supported");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "criss_cross_attention_custom: only float32 supported");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "criss_cross_attention_custom: x must be on NPU");
    TORCH_CHECK(out_h.device().type() == c10::DeviceType::PrivateUse1, "criss_cross_attention_custom: out_h must be on NPU");
    TORCH_CHECK(out_w.device().type() == c10::DeviceType::PrivateUse1, "criss_cross_attention_custom: out_w must be on NPU");
    TORCH_CHECK(gamma.device().type() == c10::DeviceType::PrivateUse1, "criss_cross_attention_custom: gamma must be on NPU");

    TORCH_CHECK(x.is_contiguous(), "criss_cross_attention_custom: x must be contiguous");
    TORCH_CHECK(out_h.is_contiguous(), "criss_cross_attention_custom: out_h must be contiguous");
    TORCH_CHECK(out_w.is_contiguous(), "criss_cross_attention_custom: out_w must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "criss_cross_attention_custom: gamma must be contiguous");

    TORCH_CHECK(x.sizes() == out_h.sizes(), "criss_cross_attention_custom: shape mismatch (x vs out_h)");
    TORCH_CHECK(x.sizes() == out_w.sizes(), "criss_cross_attention_custom: shape mismatch (x vs out_w)");
    TORCH_CHECK(gamma.numel() == 1, "criss_cross_attention_custom: gamma must be a scalar tensor (numel==1)");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnCrissCrossAttentionCustom, x, out_h, out_w, gamma, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("criss_cross_attention_custom", &criss_cross_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("criss_cross_attention_custom", &criss_cross_attention_impl_npu,
          "CrissCrossAttention fused tail: y = x + gamma*(out_h + out_w) (NPU)");
}
