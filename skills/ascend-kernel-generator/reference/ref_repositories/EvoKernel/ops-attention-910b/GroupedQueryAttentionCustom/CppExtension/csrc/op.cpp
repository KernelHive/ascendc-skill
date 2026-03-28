
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "grouped_query_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "grouped_query_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "grouped_query_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "grouped_query_attention_custom: q must be float32");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "grouped_query_attention_custom: k must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "grouped_query_attention_custom: v must be float32");

    TORCH_CHECK(q.is_contiguous(), "grouped_query_attention_custom: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "grouped_query_attention_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "grouped_query_attention_custom: v must be contiguous");

    TORCH_CHECK(q.dim() == 4, "grouped_query_attention_custom: q must be 4D [B,H,S,D]");
    TORCH_CHECK(k.dim() == 4, "grouped_query_attention_custom: k must be 4D [B,Hkv,S,D]");
    TORCH_CHECK(v.dim() == 4, "grouped_query_attention_custom: v must be 4D [B,Hkv,S,D]");

    const auto B = q.size(0);
    const auto H = q.size(1);
    const auto S = q.size(2);
    const auto D = q.size(3);

    TORCH_CHECK(B > 0 && H > 0 && S > 0 && D > 0, "grouped_query_attention_custom: empty shapes not supported");

    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "grouped_query_attention_custom: batch mismatch");
    TORCH_CHECK(k.size(2) == S && v.size(2) == S, "grouped_query_attention_custom: seq mismatch");
    TORCH_CHECK(k.size(3) == D && v.size(3) == D, "grouped_query_attention_custom: head dim mismatch");

    const auto Hkv = k.size(1);
    TORCH_CHECK(Hkv > 0, "grouped_query_attention_custom: Hkv must be > 0");
    TORCH_CHECK(v.size(1) == Hkv, "grouped_query_attention_custom: k/v Hkv mismatch");
    TORCH_CHECK(H % Hkv == 0, "grouped_query_attention_custom: H must be divisible by Hkv");

    TORCH_CHECK(D <= 128,  "grouped_query_attention_custom: head dim too large (D must be <= 128 for this kernel)");
    TORCH_CHECK(S <= 4096, "grouped_query_attention_custom: seq_len too large (S must be <= 4096 for this kernel)");
}

at::Tensor grouped_query_attention_custom_impl_npu(const at::Tensor& q,
                                                   const at::Tensor& k,
                                                   const at::Tensor& v)
{
    check_inputs(q, k, v);
    auto y = at::empty_like(q);
    EXEC_NPU_CMD(aclnnGroupedQueryAttentionCustom, q, k, v, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("grouped_query_attention_custom(Tensor q, Tensor k, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("grouped_query_attention_custom", &grouped_query_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_query_attention_custom", &grouped_query_attention_custom_impl_npu,
          "Fused Grouped Query Attention core (NPU, float32). q:[B,H,S,D], k/v:[B,Hkv,S,D] -> y:[B,H,S,D]");
}
