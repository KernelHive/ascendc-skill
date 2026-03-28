
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor double_attention_impl_npu(const at::Tensor& a,
                                    const at::Tensor& b,
                                    const at::Tensor& v)
{
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "double_attention_custom: a must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "double_attention_custom: b must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "double_attention_custom: v must be on NPU");

    TORCH_CHECK(a.scalar_type() == at::kFloat, "double_attention_custom: a must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "double_attention_custom: b must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "double_attention_custom: v must be float32");

    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && v.dim() == 3,
                "double_attention_custom: inputs must be 3D [B,C,HW]");

    TORCH_CHECK(a.is_contiguous(), "double_attention_custom: a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "double_attention_custom: b must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "double_attention_custom: v must be contiguous");

    const auto B  = a.size(0);
    const auto Cm = a.size(1);
    const auto HW = a.size(2);

    TORCH_CHECK(B > 0 && Cm > 0 && HW > 0, "double_attention_custom: invalid a shape");
    TORCH_CHECK(b.size(0) == B && v.size(0) == B, "double_attention_custom: batch mismatch");
    TORCH_CHECK(b.size(2) == HW && v.size(2) == HW, "double_attention_custom: HW mismatch");
    TORCH_CHECK(b.size(1) == v.size(1), "double_attention_custom: Cn mismatch between b and v");
    TORCH_CHECK(b.size(1) > 0, "double_attention_custom: Cn must be positive");

    at::Tensor z = at::empty({B, Cm, HW}, a.options());
    EXEC_NPU_CMD(aclnnDoubleAttentionCustom, a, b, v, z);
    return z;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("double_attention_custom", &double_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("double_attention_custom", &double_attention_impl_npu,
          "Double Attention fused core op (A2-Net): a,b,v -> z [B,Cm,HW] (NPU)");
}
