
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor par_net_attention_impl_npu(const at::Tensor& x1,
                                     const at::Tensor& x2,
                                     const at::Tensor& x3)
{
    TORCH_CHECK(x1.device().type() == c10::DeviceType::PrivateUse1, "par_net_attention_custom: x1 must be on NPU");
    TORCH_CHECK(x2.device().type() == c10::DeviceType::PrivateUse1, "par_net_attention_custom: x2 must be on NPU");
    TORCH_CHECK(x3.device().type() == c10::DeviceType::PrivateUse1, "par_net_attention_custom: x3 must be on NPU");

    TORCH_CHECK(x1.scalar_type() == at::kFloat, "par_net_attention_custom: only float32 supported");
    TORCH_CHECK(x2.scalar_type() == at::kFloat, "par_net_attention_custom: x2 must be float32");
    TORCH_CHECK(x3.scalar_type() == at::kFloat, "par_net_attention_custom: x3 must be float32");

    TORCH_CHECK(x1.dim() >= 1, "par_net_attention_custom: x1 must have rank >= 1");
    TORCH_CHECK(x2.sizes() == x1.sizes(), "par_net_attention_custom: x2 must have same shape as x1");
    TORCH_CHECK(x3.sizes() == x1.sizes(), "par_net_attention_custom: x3 must have same shape as x1");

    TORCH_CHECK(x1.is_contiguous(), "par_net_attention_custom: x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "par_net_attention_custom: x2 must be contiguous");
    TORCH_CHECK(x3.is_contiguous(), "par_net_attention_custom: x3 must be contiguous");

    at::Tensor y = at::empty_like(x1);

    // Keep naming consistent with op type: ParNetAttentionCustom -> aclnnParNetAttentionCustom
    EXEC_NPU_CMD(aclnnParNetAttentionCustom, x1, x2, x3, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("par_net_attention_custom", &par_net_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("par_net_attention_custom", &par_net_attention_impl_npu,
          "ParNetAttention fused tail on NPU: y = SiLU(x1 + x2 + x3) (float32)");
}
