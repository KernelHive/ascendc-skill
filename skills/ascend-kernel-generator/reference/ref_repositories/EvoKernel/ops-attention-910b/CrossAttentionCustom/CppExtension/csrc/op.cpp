
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"
#include <cmath>
#include <cstdint>

at::Tensor cross_attention_custom_impl_npu(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v,
                                          double scale_attr)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "cross_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "cross_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "cross_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "cross_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "cross_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "cross_attention_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "cross_attention_custom: expected 4D tensors: q [B,H,Sq,D], k/v [B,H,Sk,D]");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "cross_attention_custom: inputs must be contiguous");

    const auto B  = q.size(0);
    const auto H  = q.size(1);
    const auto Sq = q.size(2);
    const auto D  = q.size(3);
    const auto Sk = k.size(2);

    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(3) == D,
                "cross_attention_custom: k must match q in B,H,D");
    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == Sk && v.size(3) == D,
                "cross_attention_custom: v must match k shape [B,H,Sk,D]");

    TORCH_CHECK(std::isfinite(scale_attr), "cross_attention_custom: scale must be finite");

    // Keep aligned with kernel specialization to prevent silent no-write early return.
    TORCH_CHECK(D <= 64,  "cross_attention_custom: D too large for this kernel (max 64)");
    TORCH_CHECK(Sq <= 512,"cross_attention_custom: Sq too large for this kernel (max 512)");
    TORCH_CHECK(Sk <= 256,"cross_attention_custom: Sk too large for this kernel (max 256)");

    at::Tensor out = at::empty({B, H, Sq, D}, q.options());
    EXEC_NPU_CMD(aclnnCrossAttentionCustom, q, k, v, scale_attr, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cross_attention_custom", &cross_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_attention_custom", &cross_attention_custom_impl_npu,
          "Cross Attention core (QK^T*scale -> softmax -> @V) on NPU, float32");
}
