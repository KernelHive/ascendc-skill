
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor crossformer_attention_impl_npu(const at::Tensor& attn,
                                         const at::Tensor& v,
                                         const at::Tensor& proj_weight,
                                         const at::Tensor& proj_bias) {
    TORCH_CHECK(attn.device().type() == c10::DeviceType::PrivateUse1, "crossformer_attention_custom: attn must be on NPU");
    TORCH_CHECK(v.device().type()    == c10::DeviceType::PrivateUse1, "crossformer_attention_custom: v must be on NPU");
    TORCH_CHECK(proj_weight.device().type() == c10::DeviceType::PrivateUse1, "crossformer_attention_custom: proj_weight must be on NPU");
    TORCH_CHECK(proj_bias.device().type()   == c10::DeviceType::PrivateUse1, "crossformer_attention_custom: proj_bias must be on NPU");

    TORCH_CHECK(attn.scalar_type() == at::kFloat, "crossformer_attention_custom: attn must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "crossformer_attention_custom: v must be float32");
    TORCH_CHECK(proj_weight.scalar_type() == at::kFloat, "crossformer_attention_custom: proj_weight must be float32");
    TORCH_CHECK(proj_bias.scalar_type() == at::kFloat, "crossformer_attention_custom: proj_bias must be float32");

    TORCH_CHECK(attn.is_contiguous(), "crossformer_attention_custom: attn must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "crossformer_attention_custom: v must be contiguous");
    TORCH_CHECK(proj_weight.is_contiguous(), "crossformer_attention_custom: proj_weight must be contiguous");
    TORCH_CHECK(proj_bias.is_contiguous(), "crossformer_attention_custom: proj_bias must be contiguous");

    TORCH_CHECK(attn.dim() == 4, "crossformer_attention_custom: attn must be [B,H,N,N]");
    TORCH_CHECK(v.dim() == 4, "crossformer_attention_custom: v must be [B,H,N,Dh]");
    TORCH_CHECK(proj_weight.dim() == 2, "crossformer_attention_custom: proj_weight must be [C,C]");
    TORCH_CHECK(proj_bias.dim() == 1, "crossformer_attention_custom: proj_bias must be [C]");

    auto B  = attn.size(0);
    auto H  = attn.size(1);
    auto N  = attn.size(2);
    auto N2 = attn.size(3);
    TORCH_CHECK(N == N2, "crossformer_attention_custom: attn last dims must be N,N");

    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == N,
                "crossformer_attention_custom: v shape mismatch [B,H,N,Dh]");
    auto Dh = v.size(3);
    TORCH_CHECK(Dh > 0, "crossformer_attention_custom: Dh must be > 0");

    auto C = H * Dh;

    TORCH_CHECK(proj_weight.size(0) == C && proj_weight.size(1) == C,
                "crossformer_attention_custom: proj_weight must be [C,C] where C=H*Dh");
    TORCH_CHECK(proj_bias.size(0) == C,
                "crossformer_attention_custom: proj_bias must be [C] where C=H*Dh");

    at::Tensor y = at::empty({B, N, C}, attn.options());
    EXEC_NPU_CMD(aclnnCrossformerAttentionCustom, attn, v, proj_weight, proj_bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("crossformer_attention_custom", &crossformer_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crossformer_attention_custom", &crossformer_attention_impl_npu,
          "CrossFormer attention fused tail: (attn@v)->proj (NPU)");
}
