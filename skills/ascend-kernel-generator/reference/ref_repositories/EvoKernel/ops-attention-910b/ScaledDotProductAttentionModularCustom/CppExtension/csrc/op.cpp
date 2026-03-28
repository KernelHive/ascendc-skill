
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor scaled_dot_product_attention_modular_custom_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "scaled_dot_product_attention_modular_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "scaled_dot_product_attention_modular_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "scaled_dot_product_attention_modular_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "scaled_dot_product_attention_modular_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "scaled_dot_product_attention_modular_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "scaled_dot_product_attention_modular_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "scaled_dot_product_attention_modular_custom: expected 4D tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "scaled_dot_product_attention_modular_custom: inputs must be contiguous");

    const auto B  = q.size(0);
    const auto H  = q.size(1);
    const auto NQ = q.size(2);
    const auto DK = q.size(3);

    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(3) == DK,
                "scaled_dot_product_attention_modular_custom: k must be [B,H,NK,Dk] matching q in B,H,Dk");
    const auto NK = k.size(2);

    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == NK,
                "scaled_dot_product_attention_modular_custom: v must be [B,H,NK,Dv] matching k in B,H,NK");
    const auto DV = v.size(3);

    TORCH_CHECK(B > 0 && H > 0 && NQ > 0 && NK > 0 && DK > 0 && DV > 0,
                "scaled_dot_product_attention_modular_custom: all dims must be >0");

    TORCH_CHECK(NQ <= 512, "scaled_dot_product_attention_modular_custom: NQ>512 not supported");
    TORCH_CHECK(NK <= 512, "scaled_dot_product_attention_modular_custom: NK>512 not supported");
    TORCH_CHECK(DK <= 128, "scaled_dot_product_attention_modular_custom: Dk>128 not supported");
    TORCH_CHECK(DV <= 128, "scaled_dot_product_attention_modular_custom: Dv>128 not supported");

    at::Tensor out = at::empty({B, H, NQ, DV}, q.options());
    EXEC_NPU_CMD(aclnnScaledDotProductAttentionModularCustom, q, k, v, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention_modular_custom", &scaled_dot_product_attention_modular_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_dot_product_attention_modular_custom",
          &scaled_dot_product_attention_modular_custom_impl_npu,
          "ScaledDotProductAttentionModularCustom on NPU (Q:[B,H,NQ,Dk], K:[B,H,NK,Dk], V:[B,H,NK,Dv]) -> O:[B,H,NQ,Dv]");
}
