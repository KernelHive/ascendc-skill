
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor triplet_attention_impl_npu(const at::Tensor& x_ch,
                                     const at::Tensor& x_cw,
                                     const at::Tensor& x_hw)
{
    TORCH_CHECK(x_ch.scalar_type() == at::kFloat, "triplet_attention_custom: only float32 supported");
    TORCH_CHECK(x_cw.scalar_type() == at::kFloat, "triplet_attention_custom: only float32 supported");
    TORCH_CHECK(x_hw.scalar_type() == at::kFloat, "triplet_attention_custom: only float32 supported");

    TORCH_CHECK(x_ch.device().type() == c10::DeviceType::PrivateUse1, "triplet_attention_custom: x_ch must be on NPU");
    TORCH_CHECK(x_cw.device().type() == c10::DeviceType::PrivateUse1, "triplet_attention_custom: x_cw must be on NPU");
    TORCH_CHECK(x_hw.device().type() == c10::DeviceType::PrivateUse1, "triplet_attention_custom: x_hw must be on NPU");

    TORCH_CHECK(x_ch.is_contiguous(), "triplet_attention_custom: x_ch must be contiguous (ND)");
    TORCH_CHECK(x_cw.is_contiguous(), "triplet_attention_custom: x_cw must be contiguous (ND)");
    TORCH_CHECK(x_hw.is_contiguous(), "triplet_attention_custom: x_hw must be contiguous (ND)");

    TORCH_CHECK(x_ch.sizes() == x_cw.sizes(), "triplet_attention_custom: shape mismatch (x_ch vs x_cw)");
    TORCH_CHECK(x_ch.sizes() == x_hw.sizes(), "triplet_attention_custom: shape mismatch (x_ch vs x_hw)");
    TORCH_CHECK(x_ch.numel() > 0, "triplet_attention_custom: empty input not supported");

    at::Tensor y = at::empty_like(x_hw);
    EXEC_NPU_CMD(aclnnTripletAttentionCustom, x_ch, x_cw, x_hw, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("triplet_attention_custom", &triplet_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triplet_attention_custom", &triplet_attention_impl_npu,
          "Triplet Attention fused aggregation on NPU: y = (x_ch + x_cw + x_hw) / 3 (float32)");
}
