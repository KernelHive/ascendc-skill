
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q,
                         const at::Tensor& k_cache,
                         const at::Tensor& v_cache,
                         const at::Tensor& cache_seqlens,
                         const at::Tensor& page_table,
                         const at::Tensor& causal_flag)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "paged_attention_kv_cache_custom: q must be on NPU");
    TORCH_CHECK(k_cache.device().type() == c10::DeviceType::PrivateUse1, "paged_attention_kv_cache_custom: k_cache must be on NPU");
    TORCH_CHECK(v_cache.device().type() == c10::DeviceType::PrivateUse1, "paged_attention_kv_cache_custom: v_cache must be on NPU");
    TORCH_CHECK(cache_seqlens.device().type() == c10::DeviceType::PrivateUse1, "paged_attention_kv_cache_custom: cache_seqlens must be on NPU");
    TORCH_CHECK(page_table.device().type() == c10::DeviceType::PrivateUse1, "paged_attention_kv_cache_custom: page_table must be on NPU");
    TORCH_CHECK(causal_flag.device().type() == c10::DeviceType::PrivateUse1, "paged_attention_kv_cache_custom: causal_flag must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "paged_attention_kv_cache_custom: q must be bfloat16");
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16, "paged_attention_kv_cache_custom: k_cache must be bfloat16");
    TORCH_CHECK(v_cache.scalar_type() == at::kBFloat16, "paged_attention_kv_cache_custom: v_cache must be bfloat16");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "paged_attention_kv_cache_custom: cache_seqlens must be int32");
    TORCH_CHECK(page_table.scalar_type() == at::kInt, "paged_attention_kv_cache_custom: page_table must be int32");
    TORCH_CHECK(causal_flag.scalar_type() == at::kInt, "paged_attention_kv_cache_custom: causal_flag must be int32");

    TORCH_CHECK(q.is_contiguous(), "paged_attention_kv_cache_custom: q must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "paged_attention_kv_cache_custom: k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "paged_attention_kv_cache_custom: v_cache must be contiguous");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "paged_attention_kv_cache_custom: cache_seqlens must be contiguous");
    TORCH_CHECK(page_table.is_contiguous(), "paged_attention_kv_cache_custom: page_table must be contiguous");
    TORCH_CHECK(causal_flag.is_contiguous(), "paged_attention_kv_cache_custom: causal_flag must be contiguous");

    TORCH_CHECK(q.dim() == 4, "paged_attention_kv_cache_custom: q must be [B,Sq,Hq,D]");
    TORCH_CHECK(k_cache.dim() == 4, "paged_attention_kv_cache_custom: k_cache must be [NB,PBS,Hkv,D]");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "paged_attention_kv_cache_custom: v_cache shape must match k_cache");
    TORCH_CHECK(cache_seqlens.dim() == 1, "paged_attention_kv_cache_custom: cache_seqlens must be [B]");
    TORCH_CHECK(page_table.dim() == 2, "paged_attention_kv_cache_custom: page_table must be [B,MBS]");
    TORCH_CHECK(causal_flag.numel() == 1, "paged_attention_kv_cache_custom: causal_flag must have 1 element");

    TORCH_CHECK(cache_seqlens.size(0) == q.size(0), "paged_attention_kv_cache_custom: cache_seqlens B must match q");
    TORCH_CHECK(page_table.size(0) == q.size(0), "paged_attention_kv_cache_custom: page_table B must match q");
    TORCH_CHECK(k_cache.size(3) == q.size(3), "paged_attention_kv_cache_custom: head dim mismatch (k_cache D == q D)");
    TORCH_CHECK(q.size(2) % k_cache.size(2) == 0, "paged_attention_kv_cache_custom: Hq must be divisible by Hkv (GQA)");
}

at::Tensor paged_attention_kv_cache_custom_impl_npu(const at::Tensor& q,
                                                   const at::Tensor& k_cache,
                                                   const at::Tensor& v_cache,
                                                   const at::Tensor& cache_seqlens,
                                                   const at::Tensor& page_table,
                                                   const at::Tensor& causal_flag)
{
    check_inputs(q, k_cache, v_cache, cache_seqlens, page_table, causal_flag);
    at::Tensor out = at::empty_like(q);
    EXEC_NPU_CMD(aclnnPagedAttentionKVCacheCustom,
                 q, k_cache, v_cache, cache_seqlens, page_table, causal_flag,
                 out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("paged_attention_kv_cache_custom", &paged_attention_kv_cache_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_kv_cache_custom", &paged_attention_kv_cache_custom_impl_npu,
          "PagedAttentionKVCacheCustom (NPU, bf16, decoding-oriented paged KV cache attention)");
}
