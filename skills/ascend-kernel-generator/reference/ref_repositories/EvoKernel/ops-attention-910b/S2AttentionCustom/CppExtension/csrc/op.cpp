
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static at::Tensor s2_attention_impl_npu(const at::Tensor& attn,
                                       const at::Tensor& x_all)
{
    TORCH_CHECK(attn.device().type() == c10::DeviceType::PrivateUse1, "s2_attention_custom: attn must be on NPU");
    TORCH_CHECK(x_all.device().type() == c10::DeviceType::PrivateUse1, "s2_attention_custom: x_all must be on NPU");

    TORCH_CHECK(attn.scalar_type() == at::kFloat, "s2_attention_custom: attn must be float32");
    TORCH_CHECK(x_all.scalar_type() == at::kFloat, "s2_attention_custom: x_all must be float32");

    TORCH_CHECK(attn.is_contiguous(), "s2_attention_custom: attn must be contiguous");
    TORCH_CHECK(x_all.is_contiguous(), "s2_attention_custom: x_all must be contiguous");

    TORCH_CHECK(attn.dim() == 3, "s2_attention_custom: attn must be 3D [B,3,C]");
    TORCH_CHECK(x_all.dim() == 5, "s2_attention_custom: x_all must be 5D [B,3,H,W,C]");

    auto B = x_all.size(0);
    auto K = x_all.size(1);
    auto H = x_all.size(2);
    auto W = x_all.size(3);
    auto C = x_all.size(4);

    TORCH_CHECK(K == 3, "s2_attention_custom: K must be 3");
    TORCH_CHECK(attn.size(0) == B && attn.size(1) == K && attn.size(2) == C,
                "s2_attention_custom: attn shape must be [B,3,C] matching x_all");

    at::Tensor y = at::empty({B, H, W, C}, x_all.options());
    EXEC_NPU_CMD(aclnnS2AttentionCustom, attn, x_all, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("s2_attention_custom", &s2_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("s2_attention_custom", &s2_attention_impl_npu, "S2Attention fused weighted sum (NPU, float32)");
}
