
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static at::Tensor optimized_flash_attention_custom_impl_npu(const at::Tensor& q,
                                                            const at::Tensor& k,
                                                            const at::Tensor& v,
                                                            const c10::optional<at::Tensor>& bias_opt,
                                                            bool has_bias) {
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "optimized_flash_attention_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "optimized_flash_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "optimized_flash_attention_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "optimized_flash_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "optimized_flash_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "optimized_flash_attention_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "optimized_flash_attention_custom: expected q/k/v as 4D [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "optimized_flash_attention_custom: q/k/v must be contiguous");

    const auto B  = q.size(0);
    const auto H  = q.size(1);
    const auto Sq = q.size(2);
    const auto D  = q.size(3);

    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(3) == D,
                "optimized_flash_attention_custom: k must be [B,H,Sk,D] with same B,H,D as q");
    const auto Sk = k.size(2);

    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == Sk && v.size(3) == D,
                "optimized_flash_attention_custom: v must be [B,H,Sk,D] (requires d_v == d_k)");

    at::Tensor bias;
    if (has_bias) {
        TORCH_CHECK(bias_opt.has_value() && bias_opt.value().defined(),
                    "optimized_flash_attention_custom: has_bias=True requires bias tensor");
        bias = bias_opt.value();
        TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1,
                    "optimized_flash_attention_custom: bias must be on NPU");
        TORCH_CHECK(bias.scalar_type() == at::kFloat,
                    "optimized_flash_attention_custom: bias must be float32");
        TORCH_CHECK(bias.is_contiguous(),
                    "optimized_flash_attention_custom: bias must be contiguous");

        if (bias.dim() == 3) {
            TORCH_CHECK(bias.size(0) == H && bias.size(1) == Sq && bias.size(2) == Sk,
                        "optimized_flash_attention_custom: bias 3D must be [H,Sq,Sk]");
            // Do not expand+copy; kernel supports broadcast over B.
        } else if (bias.dim() == 4) {
            TORCH_CHECK(bias.size(0) == B && bias.size(1) == H && bias.size(2) == Sq && bias.size(3) == Sk,
                        "optimized_flash_attention_custom: bias 4D must be [B,H,Sq,Sk]");
        } else {
            TORCH_CHECK(false, "optimized_flash_attention_custom: bias must be 3D [H,Sq,Sk] or 4D [B,H,Sq,Sk]");
        }
    } else {
        bias = at::empty({1}, q.options());
    }

    at::Tensor out = at::empty({B, H, Sq, D}, q.options());
    EXEC_NPU_CMD(aclnnOptimizedFlashAttentionCustom, q, k, v, bias, has_bias, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("optimized_flash_attention_custom", &optimized_flash_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimized_flash_attention_custom", &optimized_flash_attention_custom_impl_npu,
          "Optimized Flash Attention (online softmax, tiled K/V loads) on NPU, float32, optional bias via has_bias");
}
