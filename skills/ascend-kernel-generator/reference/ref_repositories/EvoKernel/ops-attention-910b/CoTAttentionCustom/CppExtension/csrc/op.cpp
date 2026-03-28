
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor co_t_attention_impl_npu(const at::Tensor& k1,
                                  const at::Tensor& att,
                                  const at::Tensor& v) {
    TORCH_CHECK(k1.device().type() == c10::DeviceType::PrivateUse1, "co_t_attention_custom: k1 must be on NPU");
    TORCH_CHECK(att.device().type() == c10::DeviceType::PrivateUse1, "co_t_attention_custom: att must be on NPU");
    TORCH_CHECK(v.device().type()  == c10::DeviceType::PrivateUse1, "co_t_attention_custom: v must be on NPU");

    TORCH_CHECK(k1.scalar_type() == at::kFloat, "co_t_attention_custom: k1 must be float32");
    TORCH_CHECK(att.scalar_type() == at::kFloat, "co_t_attention_custom: att must be float32");
    TORCH_CHECK(v.scalar_type()  == at::kFloat, "co_t_attention_custom: v must be float32");

    TORCH_CHECK(k1.is_contiguous(), "co_t_attention_custom: k1 must be contiguous");
    TORCH_CHECK(att.is_contiguous(), "co_t_attention_custom: att must be contiguous");
    TORCH_CHECK(v.is_contiguous(),  "co_t_attention_custom: v must be contiguous");

    TORCH_CHECK(k1.dim() == 4, "co_t_attention_custom: k1 must be 4D [bs,C,H,W]");
    TORCH_CHECK(att.dim() == 3, "co_t_attention_custom: att must be 3D [bs,C,hw]");
    TORCH_CHECK(v.dim() == 3, "co_t_attention_custom: v must be 3D [bs,C,hw]");

    auto bs = k1.size(0);
    auto C  = k1.size(1);
    auto H  = k1.size(2);
    auto W  = k1.size(3);
    auto hw = H * W;

    TORCH_CHECK(att.size(0) == bs && att.size(1) == C && att.size(2) == hw,
                "co_t_attention_custom: att shape must be [bs,C,H*W]");
    TORCH_CHECK(v.size(0) == bs && v.size(1) == C && v.size(2) == hw,
                "co_t_attention_custom: v shape must be [bs,C,H*W]");

    at::Tensor y = at::empty_like(k1);
    EXEC_NPU_CMD(aclnnCoTAttentionCustom, k1, att, v, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("co_t_attention_custom", &co_t_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("co_t_attention_custom", &co_t_attention_impl_npu, "CoTAttention tail fused (softmax*V + K1) (NPU)");
}
