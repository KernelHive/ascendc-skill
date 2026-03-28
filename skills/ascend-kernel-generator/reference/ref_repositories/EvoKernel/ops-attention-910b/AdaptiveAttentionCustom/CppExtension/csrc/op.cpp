
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "adaptive_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "adaptive_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "adaptive_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "adaptive_attention_custom: q must be float32");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "adaptive_attention_custom: k must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "adaptive_attention_custom: v must be float32");

    TORCH_CHECK(q.is_contiguous(), "adaptive_attention_custom: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "adaptive_attention_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "adaptive_attention_custom: v must be contiguous");

    TORCH_CHECK(q.dim() == 4, "adaptive_attention_custom: q must be 4D [B,H,S,D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "adaptive_attention_custom: k shape must match q [B,H,S,D]");
    TORCH_CHECK(v.sizes() == q.sizes(), "adaptive_attention_custom: v shape must match q [B,H,S,D]");

    const auto B = q.size(0);
    const auto H = q.size(1);
    const auto S = q.size(2);
    const auto D = q.size(3);
    TORCH_CHECK(B > 0 && H > 0 && S > 0 && D > 0, "adaptive_attention_custom: empty shapes not supported");

    TORCH_CHECK(D <= 128,  "adaptive_attention_custom: head dim too large (D must be <= 128 for this kernel)");
    TORCH_CHECK(S <= 4096, "adaptive_attention_custom: seq_len too large (S must be <= 4096 for this kernel)");
}

at::Tensor adaptive_attention_custom_impl_npu(const at::Tensor& q,
                                             const at::Tensor& k,
                                             const at::Tensor& v)
{
    check_inputs(q, k, v);
    auto y = at::empty_like(q);
    EXEC_NPU_CMD(aclnnAdaptiveAttentionCustom, q, k, v, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("adaptive_attention_custom(Tensor q, Tensor k, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("adaptive_attention_custom", &adaptive_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_attention_custom", &adaptive_attention_custom_impl_npu,
          "Fused scaled dot-product attention (NPU, float32). q,k,v:[B,H,S,D] -> y:[B,H,S,D]");
}
