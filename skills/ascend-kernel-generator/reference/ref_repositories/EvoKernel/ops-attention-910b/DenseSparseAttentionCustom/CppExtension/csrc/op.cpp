
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& kv_cache,
                         const at::Tensor& indices)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "dense_sparse_attention_custom: q must be on NPU");
    TORCH_CHECK(kv_cache.device().type() == c10::DeviceType::PrivateUse1, "dense_sparse_attention_custom: kv_cache must be on NPU");
    TORCH_CHECK(indices.device().type() == c10::DeviceType::PrivateUse1, "dense_sparse_attention_custom: indices must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "dense_sparse_attention_custom: q must be bfloat16");
    TORCH_CHECK(kv_cache.scalar_type() == at::kBFloat16, "dense_sparse_attention_custom: kv_cache must be bfloat16");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "dense_sparse_attention_custom: indices must be int32");

    TORCH_CHECK(q.is_contiguous(), "dense_sparse_attention_custom: q must be contiguous");
    TORCH_CHECK(kv_cache.is_contiguous(), "dense_sparse_attention_custom: kv_cache must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "dense_sparse_attention_custom: indices must be contiguous");

    TORCH_CHECK(q.dim() == 4, "dense_sparse_attention_custom: q must be [B,Sq,H,Dqk]");
    TORCH_CHECK(kv_cache.dim() == 4, "dense_sparse_attention_custom: kv_cache must be [NB,PBS,1,Dqk]");
    TORCH_CHECK(indices.dim() == 3, "dense_sparse_attention_custom: indices must be [B,Sq,topk]");

    const auto B = q.size(0);
    TORCH_CHECK(indices.size(0) == B, "dense_sparse_attention_custom: indices.size(0) must equal B");
    TORCH_CHECK(indices.size(1) == q.size(1), "dense_sparse_attention_custom: indices.size(1) must equal Sq");

    TORCH_CHECK(kv_cache.size(2) == 1, "dense_sparse_attention_custom: kv_cache.size(2) must be 1");
    TORCH_CHECK(kv_cache.size(3) == q.size(3), "dense_sparse_attention_custom: Dqk mismatch (kv_cache last dim must equal q last dim)");

    TORCH_CHECK(q.size(1) == 1, "dense_sparse_attention_custom: this kernel supports decoding only (Sq=1)");
    TORCH_CHECK(q.size(3) == 576, "dense_sparse_attention_custom: this kernel supports Dqk=576 only");
    TORCH_CHECK(kv_cache.size(1) == 16, "dense_sparse_attention_custom: this kernel supports page_block_size=16 only");
    TORCH_CHECK(indices.size(2) > 0 && indices.size(2) <= 32, "dense_sparse_attention_custom: this kernel supports 1<=topk<=32 only");
}

at::Tensor dense_sparse_attention_custom_impl_npu(const at::Tensor& q,
                                                  const at::Tensor& kv_cache,
                                                  const at::Tensor& indices)
{
    check_inputs(q, kv_cache, indices);
    auto out = at::empty({q.size(0), q.size(1), q.size(2), 512}, q.options());
    EXEC_NPU_CMD(aclnnDenseSparseAttentionCustom, q, kv_cache, indices, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("dense_sparse_attention_custom", &dense_sparse_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_sparse_attention_custom", &dense_sparse_attention_custom_impl_npu,
          "DenseSparseAttentionCustom (NPU, bf16, decoding-focused DSA)");
}
