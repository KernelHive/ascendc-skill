
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor strided_attention_custom_impl_npu(const at::Tensor& q,
                                            const at::Tensor& k,
                                            const at::Tensor& v,
                                            int64_t stride) {
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "strided_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "strided_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "strided_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "strided_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "strided_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "strided_attention_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "strided_attention_custom: expected 4D tensors [B,H,S,D]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "strided_attention_custom: q/k/v must have identical shapes [B,H,S,D]");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "strided_attention_custom: q/k/v must be contiguous");

    TORCH_CHECK(stride > 0, "strided_attention_custom: stride must be > 0");

    const auto S = q.size(2);
    const auto D = q.size(3);

    TORCH_CHECK(S > 0 && S <= 512, "strided_attention_custom: only supports 1 <= S <= 512");
    TORCH_CHECK(D > 0 && D <= 64,  "strided_attention_custom: only supports 1 <= D <= 64");
    TORCH_CHECK(stride <= S, "strided_attention_custom: stride must be <= S");

    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnStridedAttentionCustom, q, k, v, stride, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("strided_attention_custom", &strided_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_custom", &strided_attention_custom_impl_npu,
          "Strided attention fused core on NPU (float32, contiguous, [B,H,S,D], S<=512, D<=64)");
}
