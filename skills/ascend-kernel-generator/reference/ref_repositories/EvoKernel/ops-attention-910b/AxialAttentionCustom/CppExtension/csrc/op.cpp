
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"
#include <cmath>
#include <cstdint>

at::Tensor axial_attention_custom_impl_npu(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v,
                                          double scale_attr)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "axial_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "axial_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "axial_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "axial_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "axial_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "axial_attention_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "axial_attention_custom: expected 3D tensors [BH, T, E]");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "axial_attention_custom: inputs must be contiguous");

    const auto BH = q.size(0);
    const auto T  = q.size(1);
    const auto E  = q.size(2);

    TORCH_CHECK(k.size(0) == BH && k.size(1) == T && k.size(2) == E,
                "axial_attention_custom: k must match q shape [BH,T,E]");
    TORCH_CHECK(v.size(0) == BH && v.size(1) == T && v.size(2) == E,
                "axial_attention_custom: v must match q shape [BH,T,E]");

    TORCH_CHECK(std::isfinite(scale_attr), "axial_attention_custom: scale must be finite");

    // Must match kernel bounds; do not allow silent early-return.
    TORCH_CHECK(T <= 7,  "axial_attention_custom: T too large for this kernel (max 7)");
    TORCH_CHECK(E <= 64, "axial_attention_custom: E too large for this kernel (max 64)");

    at::Tensor out = at::empty({BH, T, E}, q.options());
    EXEC_NPU_CMD(aclnnAxialAttentionCustom, q, k, v, scale_attr, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("axial_attention_custom", &axial_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("axial_attention_custom", &axial_attention_custom_impl_npu,
          "Axial attention core on NPU: softmax((Q@K^T)*scale)@V for [BH,T,E], float32");
}
