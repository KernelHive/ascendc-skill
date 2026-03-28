
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "cosformer_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "cosformer_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "cosformer_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "cosformer_attention_custom: q must be float32");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "cosformer_attention_custom: k must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "cosformer_attention_custom: v must be float32");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "cosformer_attention_custom: expected 4D tensors [B,H,S,D]");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "cosformer_attention_custom: inputs must be contiguous");

    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "cosformer_attention_custom: q/k/v must have identical shapes [B,H,S,D]");

    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(3) > 0,
                "cosformer_attention_custom: all dimensions must be > 0");

    TORCH_CHECK(q.size(3) <= 64,   "cosformer_attention_custom: D > 64 not supported by this kernel");
    TORCH_CHECK(q.size(2) <= 1024, "cosformer_attention_custom: S > 1024 not supported by this kernel");
}

at::Tensor cosformer_attention_custom_impl_npu(const at::Tensor& q,
                                               const at::Tensor& k,
                                               const at::Tensor& v) {
    check_inputs(q, k, v);
    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnCosformerAttentionCustom, q, k, v, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cosformer_attention_custom", &cosformer_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cosformer_attention_custom", &cosformer_attention_custom_impl_npu,
          "CosformerAttentionCustom fused core on NPU (float32): q/k/v [B,H,S,D] -> out [B,H,S,D], includes relu+exp-decay, eps=1e-6");
}
