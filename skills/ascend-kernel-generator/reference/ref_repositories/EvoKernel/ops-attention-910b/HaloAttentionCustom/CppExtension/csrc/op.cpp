
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_qkv_3d(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D [B,I/J,D]");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static inline void check_mask(const at::Tensor& t) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "halo_attention_custom: mask must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kBool, "halo_attention_custom: mask must be bool (True means masked)");
    TORCH_CHECK(t.dim() == 3, "halo_attention_custom: mask must be 3D [B,1,J]");
    TORCH_CHECK(t.is_contiguous(), "halo_attention_custom: mask must be contiguous");
    TORCH_CHECK(t.size(1) == 1, "halo_attention_custom: mask second dim must be 1");
}

at::Tensor halo_attention_custom_impl_npu(const at::Tensor& q,
                                         const at::Tensor& k,
                                         const at::Tensor& v,
                                         const at::Tensor& mask) {
    check_qkv_3d(q, "halo_attention_custom: q");
    check_qkv_3d(k, "halo_attention_custom: k");
    check_qkv_3d(v, "halo_attention_custom: v");
    check_mask(mask);

    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "halo_attention_custom: B mismatch");
    TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "halo_attention_custom: D mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "halo_attention_custom: J mismatch between k and v");
    TORCH_CHECK(mask.size(0) == q.size(0) && mask.size(2) == k.size(1), "halo_attention_custom: mask must be [B,1,J] matching B and J");
    TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 && q.size(2) > 0, "halo_attention_custom: I,J,D must be >0");

    at::Tensor out = at::empty({q.size(0), q.size(1), q.size(2)}, q.options());
    EXEC_NPU_CMD(aclnnHaloAttentionCustom, q, k, v, mask, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("halo_attention_custom", &halo_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("halo_attention_custom", &halo_attention_custom_impl_npu,
          "Halo attention fused core on NPU (QK^T + masked_fill + softmax + AV), float32; mask bool True=masked");
}
