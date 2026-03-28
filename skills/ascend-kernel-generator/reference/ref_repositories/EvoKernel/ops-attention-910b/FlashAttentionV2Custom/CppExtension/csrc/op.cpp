
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v,
                         const at::Tensor& scale)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "flash_attention_v2_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "flash_attention_v2_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "flash_attention_v2_custom: v must be on NPU");
    TORCH_CHECK(scale.device().type() == c10::DeviceType::PrivateUse1, "flash_attention_v2_custom: scale must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "flash_attention_v2_custom: q must be float32");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "flash_attention_v2_custom: k must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "flash_attention_v2_custom: v must be float32");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "flash_attention_v2_custom: scale must be float32");

    TORCH_CHECK(q.is_contiguous(), "flash_attention_v2_custom: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "flash_attention_v2_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "flash_attention_v2_custom: v must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "flash_attention_v2_custom: scale must be contiguous");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "flash_attention_v2_custom: q/k/v must be [B,H,S,D]");
    TORCH_CHECK(scale.dim() == 1 && scale.numel() == 1, "flash_attention_v2_custom: scale must be scalar tensor [1]");

    auto B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(2) == S && k.size(3) == D, "flash_attention_v2_custom: k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == S && v.size(3) == D, "flash_attention_v2_custom: v shape mismatch");
}

at::Tensor flash_attention_v2_impl_npu(const at::Tensor& q,
                                      const at::Tensor& k,
                                      const at::Tensor& v,
                                      const at::Tensor& scale)
{
    check_inputs(q, k, v, scale);
    auto o = at::empty_like(q);
    EXEC_NPU_CMD(aclnnFlashAttentionV2Custom, q, k, v, scale, o);
    return o;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("flash_attention_v2_custom", &flash_attention_v2_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_v2_custom", &flash_attention_v2_impl_npu,
          "Fused attention v2: softmax((QK^T)*scale)@V (NPU, FP32, [B,H,S,D])");
}
