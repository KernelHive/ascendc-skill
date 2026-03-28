
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor sk_attention_impl_npu(const at::Tensor& attn, const at::Tensor& feats) {
    // attn:  [K, bs, C, 1, 1] float32 contiguous
    // feats: [K, bs, C, H, W] float32 contiguous
    TORCH_CHECK(attn.device().type() == c10::DeviceType::PrivateUse1, "sk_attention_custom: attn must be on NPU");
    TORCH_CHECK(feats.device().type() == c10::DeviceType::PrivateUse1, "sk_attention_custom: feats must be on NPU");
    TORCH_CHECK(attn.scalar_type() == at::kFloat, "sk_attention_custom: attn must be float32");
    TORCH_CHECK(feats.scalar_type() == at::kFloat, "sk_attention_custom: feats must be float32");
    TORCH_CHECK(attn.is_contiguous(), "sk_attention_custom: attn must be contiguous");
    TORCH_CHECK(feats.is_contiguous(), "sk_attention_custom: feats must be contiguous");

    TORCH_CHECK(attn.dim() == 5, "sk_attention_custom: attn must be 5D [K,bs,C,1,1]");
    TORCH_CHECK(feats.dim() == 5, "sk_attention_custom: feats must be 5D [K,bs,C,H,W]");

    auto K  = feats.size(0);
    auto bs = feats.size(1);
    auto C  = feats.size(2);
    auto H  = feats.size(3);
    auto W  = feats.size(4);

    TORCH_CHECK(attn.size(0) == K,  "sk_attention_custom: attn K mismatch");
    TORCH_CHECK(attn.size(1) == bs, "sk_attention_custom: attn bs mismatch");
    TORCH_CHECK(attn.size(2) == C,  "sk_attention_custom: attn C mismatch");
    TORCH_CHECK(attn.size(3) == 1 && attn.size(4) == 1, "sk_attention_custom: attn last dims must be 1,1");

    at::Tensor y = at::empty({bs, C, H, W}, feats.options());
    EXEC_NPU_CMD(aclnnSKAttentionCustom, attn, feats, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sk_attention_custom", &sk_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sk_attention_custom", &sk_attention_impl_npu, "SK attention fused weighted reduction (NPU)");
}
