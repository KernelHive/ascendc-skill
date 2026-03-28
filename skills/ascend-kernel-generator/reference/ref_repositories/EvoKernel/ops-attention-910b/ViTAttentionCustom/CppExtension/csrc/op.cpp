
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor vi_t_attention_custom_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& scale,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "vi_t_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "vi_t_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "vi_t_attention_custom: v must be on NPU");
    TORCH_CHECK(scale.device().type() == c10::DeviceType::PrivateUse1, "vi_t_attention_custom: scale must be on NPU");
    TORCH_CHECK(proj_weight.device().type() == c10::DeviceType::PrivateUse1, "vi_t_attention_custom: proj_weight must be on NPU");
    TORCH_CHECK(proj_bias.device().type() == c10::DeviceType::PrivateUse1, "vi_t_attention_custom: proj_bias must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "vi_t_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "vi_t_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "vi_t_attention_custom: only float32 supported");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "vi_t_attention_custom: only float32 supported");
    TORCH_CHECK(proj_weight.scalar_type() == at::kFloat, "vi_t_attention_custom: only float32 supported");
    TORCH_CHECK(proj_bias.scalar_type() == at::kFloat, "vi_t_attention_custom: only float32 supported");

    TORCH_CHECK(q.is_contiguous(), "vi_t_attention_custom: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "vi_t_attention_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "vi_t_attention_custom: v must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "vi_t_attention_custom: scale must be contiguous");
    TORCH_CHECK(proj_weight.is_contiguous(), "vi_t_attention_custom: proj_weight must be contiguous");
    TORCH_CHECK(proj_bias.is_contiguous(), "vi_t_attention_custom: proj_bias must be contiguous");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "vi_t_attention_custom: q/k/v must be [bs,heads,nq,d]");
    TORCH_CHECK(scale.numel() == 1, "vi_t_attention_custom: scale must be a 1-element tensor");
    TORCH_CHECK(proj_weight.dim() == 2, "vi_t_attention_custom: proj_weight must be [C,C]");
    TORCH_CHECK(proj_bias.dim() == 1, "vi_t_attention_custom: proj_bias must be [C]");

    TORCH_CHECK(k.sizes() == q.sizes(), "vi_t_attention_custom: k shape must equal q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "vi_t_attention_custom: v shape must equal q shape");

    auto bs = q.size(0);
    auto heads = q.size(1);
    auto nq = q.size(2);
    auto d = q.size(3);
    auto c = heads * d;

    TORCH_CHECK(proj_weight.size(0) == c && proj_weight.size(1) == c, "vi_t_attention_custom: proj_weight must be [C,C] with C=heads*d");
    TORCH_CHECK(proj_bias.size(0) == c, "vi_t_attention_custom: proj_bias must be [C] with C=heads*d");

    at::Tensor y = at::empty({bs, nq, c}, q.options());
    EXEC_NPU_CMD(aclnnViTAttentionCustom, q, k, v, scale, proj_weight, proj_bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("vi_t_attention_custom", &vi_t_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vi_t_attention_custom", &vi_t_attention_custom_impl_npu, "Fused ViT attention (q,k,v,scale -> proj) on NPU");
}
