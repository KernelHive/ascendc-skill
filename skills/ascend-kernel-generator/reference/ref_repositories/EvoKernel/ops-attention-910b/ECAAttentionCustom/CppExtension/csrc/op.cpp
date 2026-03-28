
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void eca_check_weight(const at::Tensor& weight) {
    TORCH_CHECK(weight.dim() == 3, "eca_attention_custom: weight must be [1,1,K]");
    TORCH_CHECK(weight.size(0) == 1 && weight.size(1) == 1, "eca_attention_custom: weight must be [1,1,K]");
    TORCH_CHECK(weight.size(2) > 0, "eca_attention_custom: K must be > 0");
    TORCH_CHECK((weight.size(2) % 2) == 1, "eca_attention_custom: K must be odd for same padding");
    TORCH_CHECK(weight.is_contiguous(), "eca_attention_custom: weight must be contiguous");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "eca_attention_custom: only float32 supported");
}

at::Tensor eca_attention_custom_impl_npu(const at::Tensor& x,
                                        const at::Tensor& weight) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "eca_attention_custom: x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "eca_attention_custom: weight must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "eca_attention_custom: only float32 supported");
    eca_check_weight(weight);

    TORCH_CHECK(x.dim() == 4, "eca_attention_custom: x must be [B,C,H,W]");
    TORCH_CHECK(x.is_contiguous(), "eca_attention_custom: x must be contiguous NCHW");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0,
                "eca_attention_custom: all x dims must be > 0");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnECAAttentionCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("eca_attention_custom", &eca_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eca_attention_custom", &eca_attention_custom_impl_npu,
          "ECAAttentionCustom fused op: y = x * sigmoid(conv1d(mean_hw(x))) (NPU, float32)");
}
