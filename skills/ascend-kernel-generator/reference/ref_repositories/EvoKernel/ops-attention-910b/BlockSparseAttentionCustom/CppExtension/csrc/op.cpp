
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& k,
                         const at::Tensor& v,
                         const at::Tensor& scale)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "block_sparse_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "block_sparse_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "block_sparse_attention_custom: v must be on NPU");
    TORCH_CHECK(scale.device().type() == c10::DeviceType::PrivateUse1, "block_sparse_attention_custom: scale must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "block_sparse_attention_custom: q must be float32");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "block_sparse_attention_custom: k must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "block_sparse_attention_custom: v must be float32");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "block_sparse_attention_custom: scale must be float32");

    TORCH_CHECK(q.is_contiguous(), "block_sparse_attention_custom: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "block_sparse_attention_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "block_sparse_attention_custom: v must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "block_sparse_attention_custom: scale must be contiguous");

    TORCH_CHECK(q.dim() == 5, "block_sparse_attention_custom: q must be 5D [B,H,NB,BS,DK]");
    TORCH_CHECK(k.sizes() == q.sizes(), "block_sparse_attention_custom: k shape must match q");
    TORCH_CHECK(v.sizes() == q.sizes(), "block_sparse_attention_custom: v shape must match q");

    TORCH_CHECK(scale.numel() == 1, "block_sparse_attention_custom: scale must be 1 element");
    TORCH_CHECK(q.size(3) > 0 && q.size(4) > 0, "block_sparse_attention_custom: BS and DK must be > 0");
}

at::Tensor block_sparse_attention_impl_npu(const at::Tensor& q,
                                          const at::Tensor& k,
                                          const at::Tensor& v,
                                          const at::Tensor& scale)
{
    check_inputs(q, k, v, scale);
    auto y = at::empty_like(q);
    EXEC_NPU_CMD(aclnnBlockSparseAttentionCustom, q, k, v, scale, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("block_sparse_attention_custom", &block_sparse_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_attention_custom", &block_sparse_attention_impl_npu,
          "Block sparse intra-block attention fused (NPU)");
}
