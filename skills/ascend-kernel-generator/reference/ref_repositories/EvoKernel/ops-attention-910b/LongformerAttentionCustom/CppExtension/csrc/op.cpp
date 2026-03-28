
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_qkv(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D [B,H,S,D]");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor longformer_attention_custom_impl_npu(const at::Tensor& q,
                                               const at::Tensor& k,
                                               const at::Tensor& v,
                                               int64_t window_size) {
    check_qkv(q, "longformer_attention_custom: q");
    check_qkv(k, "longformer_attention_custom: k");
    check_qkv(v, "longformer_attention_custom: v");

    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "longformer_attention_custom: q/k/v shapes must match exactly");

    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    TORCH_CHECK(window_size > 0, "longformer_attention_custom: window_size must be > 0");
    TORCH_CHECK(window_size <= S, "longformer_attention_custom: window_size must be <= sequence length");

    // Semantic parity with reference model's fixed global_attention_indices=[0,511]
    TORCH_CHECK(S == 512, "longformer_attention_custom: this fused kernel matches reference globals [0,511] and requires S==512");

    // Kernel envelope (UB sizing)
    TORCH_CHECK(D > 0 && D <= 64, "longformer_attention_custom: only head_dim D<=64 supported");
    TORCH_CHECK(window_size <= 32, "longformer_attention_custom: only window_size<=32 supported");

    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnLongformerAttentionCustom, q, k, v, window_size, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("longformer_attention_custom", &longformer_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("longformer_attention_custom", &longformer_attention_custom_impl_npu,
          "Longformer attention core on NPU (row-parallel; local window + global keys {0,511}; global queries attend all), float32, [B,H,S,D]");
}
