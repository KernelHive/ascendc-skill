
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_npu_f32_contig_nonempty(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), "mobile_vi_t_attention_custom: ", name, " is undefined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "mobile_vi_t_attention_custom: ", name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, "mobile_vi_t_attention_custom: ", name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), "mobile_vi_t_attention_custom: ", name, " must be contiguous (ND)");
    TORCH_CHECK(t.numel() > 0, "mobile_vi_t_attention_custom: ", name, " must be non-empty");
}

at::Tensor mobile_vi_t_attention_custom_impl_npu(const at::Tensor& x,
                                                const at::Tensor& att_delta,
                                                const at::Tensor& ffn_delta)
{
    check_npu_f32_contig_nonempty(x, "x");
    check_npu_f32_contig_nonempty(att_delta, "att_delta");
    check_npu_f32_contig_nonempty(ffn_delta, "ffn_delta");

    TORCH_CHECK(x.sizes() == att_delta.sizes(), "mobile_vi_t_attention_custom: shape mismatch (x vs att_delta)");
    TORCH_CHECK(x.sizes() == ffn_delta.sizes(), "mobile_vi_t_attention_custom: shape mismatch (x vs ffn_delta)");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnMobileViTAttentionCustom, x, att_delta, ffn_delta, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mobile_vi_t_attention_custom", &mobile_vi_t_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mobile_vi_t_attention_custom", &mobile_vi_t_attention_custom_impl_npu,
          "MobileViT fused residual add: y = x + att_delta + ffn_delta (NPU)");
}
