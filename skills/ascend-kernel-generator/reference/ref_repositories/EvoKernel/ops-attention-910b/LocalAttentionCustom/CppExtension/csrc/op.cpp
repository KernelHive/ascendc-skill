
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_qkv(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D [B,H,S,D]");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor local_attention_custom_impl_npu(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v,
                                          int64_t window_size) {
    check_qkv(q, "local_attention_custom: q");
    check_qkv(k, "local_attention_custom: k");
    check_qkv(v, "local_attention_custom: v");

    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "local_attention_custom: q/k/v shapes must match exactly [B,H,S,D]");

    TORCH_CHECK(window_size > 0, "local_attention_custom: window_size must be > 0");
    const int64_t S = q.size(2);
    TORCH_CHECK(S > 0, "local_attention_custom: S must be > 0");
    if (window_size > S) window_size = S;

    // Kernel envelope (kept aligned with kernel caps)
    TORCH_CHECK(S <= 512, "local_attention_custom: only S<=512 supported by this kernel");
    TORCH_CHECK(q.size(3) <= 128, "local_attention_custom: only D<=128 supported by this kernel");
    TORCH_CHECK(window_size <= 128, "local_attention_custom: only window_size<=128 supported by this kernel");

    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnLocalAttentionCustom, q, k, v, window_size, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("local_attention_custom", &local_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_attention_custom", &local_attention_custom_impl_npu,
          "LocalAttentionCustom fused sliding-window attention core on NPU (Q/K/V:[B,H,S,D])->O:[B,H,S,D], float32");
}
