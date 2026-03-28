
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor sequential_polarized_self_attention_impl_npu(
    const at::Tensor& x,
    const at::Tensor& channel_weight,
    const at::Tensor& spatial_weight)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "sequential_polarized_self_attention_custom: x must be float32");
    TORCH_CHECK(channel_weight.scalar_type() == at::kFloat,
                "sequential_polarized_self_attention_custom: channel_weight must be float32");
    TORCH_CHECK(spatial_weight.scalar_type() == at::kFloat,
                "sequential_polarized_self_attention_custom: spatial_weight must be float32");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "sequential_polarized_self_attention_custom: x must be on NPU");
    TORCH_CHECK(channel_weight.device().type() == c10::DeviceType::PrivateUse1,
                "sequential_polarized_self_attention_custom: channel_weight must be on NPU");
    TORCH_CHECK(spatial_weight.device().type() == c10::DeviceType::PrivateUse1,
                "sequential_polarized_self_attention_custom: spatial_weight must be on NPU");

    TORCH_CHECK(x.is_contiguous(),
                "sequential_polarized_self_attention_custom: x must be contiguous (ND/NCHW)");
    TORCH_CHECK(channel_weight.is_contiguous(),
                "sequential_polarized_self_attention_custom: channel_weight must be contiguous (ND)");
    TORCH_CHECK(spatial_weight.is_contiguous(),
                "sequential_polarized_self_attention_custom: spatial_weight must be contiguous (ND)");

    TORCH_CHECK(x.dim() == 4, "sequential_polarized_self_attention_custom: x must be [B,C,H,W]");
    TORCH_CHECK(channel_weight.dim() == 4,
                "sequential_polarized_self_attention_custom: channel_weight must be [B,C,1,1]");
    TORCH_CHECK(spatial_weight.dim() == 4,
                "sequential_polarized_self_attention_custom: spatial_weight must be [B,1,H,W]");

    const auto B = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0,
                "sequential_polarized_self_attention_custom: invalid x shape");

    TORCH_CHECK(channel_weight.size(0) == B && channel_weight.size(1) == C &&
                channel_weight.size(2) == 1 && channel_weight.size(3) == 1,
                "sequential_polarized_self_attention_custom: channel_weight must be [B,C,1,1] matching x(B,C)");
    TORCH_CHECK(spatial_weight.size(0) == B && spatial_weight.size(1) == 1 &&
                spatial_weight.size(2) == H && spatial_weight.size(3) == W,
                "sequential_polarized_self_attention_custom: spatial_weight must be [B,1,H,W] matching x(B,H,W)");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSequentialPolarizedSelfAttentionCustom, x, channel_weight, spatial_weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sequential_polarized_self_attention_custom", &sequential_polarized_self_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sequential_polarized_self_attention_custom",
          &sequential_polarized_self_attention_impl_npu,
          "Sequential Polarized Self-Attention fused final gating: y = x*channel_weight*spatial_weight (NPU, float32)");
}
