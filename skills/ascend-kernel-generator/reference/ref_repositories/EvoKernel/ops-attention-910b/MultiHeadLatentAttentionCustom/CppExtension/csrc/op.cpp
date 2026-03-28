
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& kv_cache,
                         const at::Tensor& block_table,
                         const at::Tensor& cache_seqlens,
                         const at::Tensor& causal_flag)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "multi_head_latent_attention_custom: q must be on NPU");
    TORCH_CHECK(kv_cache.device().type() == c10::DeviceType::PrivateUse1, "multi_head_latent_attention_custom: kv_cache must be on NPU");
    TORCH_CHECK(block_table.device().type() == c10::DeviceType::PrivateUse1, "multi_head_latent_attention_custom: block_table must be on NPU");
    TORCH_CHECK(cache_seqlens.device().type() == c10::DeviceType::PrivateUse1, "multi_head_latent_attention_custom: cache_seqlens must be on NPU");
    TORCH_CHECK(causal_flag.device().type() == c10::DeviceType::PrivateUse1, "multi_head_latent_attention_custom: causal_flag must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "multi_head_latent_attention_custom: q must be bfloat16");
    TORCH_CHECK(kv_cache.scalar_type() == at::kBFloat16, "multi_head_latent_attention_custom: kv_cache must be bfloat16");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "multi_head_latent_attention_custom: block_table must be int32");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "multi_head_latent_attention_custom: cache_seqlens must be int32");
    TORCH_CHECK(causal_flag.scalar_type() == at::kInt, "multi_head_latent_attention_custom: causal_flag must be int32");

    TORCH_CHECK(q.is_contiguous(), "multi_head_latent_attention_custom: q must be contiguous");
    TORCH_CHECK(kv_cache.is_contiguous(), "multi_head_latent_attention_custom: kv_cache must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "multi_head_latent_attention_custom: block_table must be contiguous");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "multi_head_latent_attention_custom: cache_seqlens must be contiguous");
    TORCH_CHECK(causal_flag.is_contiguous(), "multi_head_latent_attention_custom: causal_flag must be contiguous");

    TORCH_CHECK(q.dim() == 4, "multi_head_latent_attention_custom: q must be [B,Sq,Hq,Dqk]");
    TORCH_CHECK(kv_cache.dim() == 4, "multi_head_latent_attention_custom: kv_cache must be [NB,PBS,1,Dqk]");
    TORCH_CHECK(block_table.dim() == 2, "multi_head_latent_attention_custom: block_table must be [B,MBS]");
    TORCH_CHECK(cache_seqlens.dim() == 1, "multi_head_latent_attention_custom: cache_seqlens must be [B]");
    TORCH_CHECK(causal_flag.numel() == 1, "multi_head_latent_attention_custom: causal_flag must have 1 element");

    const auto B = q.size(0);
    TORCH_CHECK(cache_seqlens.size(0) == B, "multi_head_latent_attention_custom: cache_seqlens.size(0) must equal B");
    TORCH_CHECK(block_table.size(0) == B, "multi_head_latent_attention_custom: block_table.size(0) must equal B");

    TORCH_CHECK(kv_cache.size(2) == 1, "multi_head_latent_attention_custom: kv_cache.size(2) must be 1");
    TORCH_CHECK(kv_cache.size(3) == q.size(3), "multi_head_latent_attention_custom: Dqk mismatch");

    // Kernel target constraints
    TORCH_CHECK(q.size(1) == 1, "multi_head_latent_attention_custom: decoding only (Sq=1)");
    TORCH_CHECK(q.size(3) == 576, "multi_head_latent_attention_custom: Dqk=576 only");
    TORCH_CHECK(kv_cache.size(1) == 16, "multi_head_latent_attention_custom: page_block_size=16 only");
}

at::Tensor multi_head_latent_attention_custom_impl_npu(const at::Tensor& q,
                                                       const at::Tensor& kv_cache,
                                                       const at::Tensor& block_table,
                                                       const at::Tensor& cache_seqlens,
                                                       const at::Tensor& causal_flag)
{
    check_inputs(q, kv_cache, block_table, cache_seqlens, causal_flag);

    auto out = at::empty({q.size(0), q.size(1), q.size(2), 512}, q.options());

    EXEC_NPU_CMD(aclnnMultiHeadLatentAttentionCustom,
                 q, kv_cache, block_table, cache_seqlens, causal_flag,
                 out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("multi_head_latent_attention_custom", &multi_head_latent_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_latent_attention_custom", &multi_head_latent_attention_custom_impl_npu,
          "MultiHeadLatentAttentionCustom (NPU, bf16, DeepSeek-V3 MLA decoding kernel)");
}
