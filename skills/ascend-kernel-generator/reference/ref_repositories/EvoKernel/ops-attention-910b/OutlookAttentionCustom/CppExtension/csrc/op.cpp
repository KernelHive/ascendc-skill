
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor outlook_attention_impl_npu(const at::Tensor& attn,
                                     const at::Tensor& v) {
    TORCH_CHECK(attn.device().type() == c10::DeviceType::PrivateUse1, "outlook_attention_custom: attn must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "outlook_attention_custom: v must be on NPU");

    TORCH_CHECK(attn.scalar_type() == at::kFloat, "outlook_attention_custom: attn must be float32");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "outlook_attention_custom: v must be float32");

    TORCH_CHECK(attn.is_contiguous(), "outlook_attention_custom: attn must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "outlook_attention_custom: v must be contiguous");

    TORCH_CHECK(attn.dim() == 5, "outlook_attention_custom: attn must be 5D [B,NH,HWo,K2,K2]");
    TORCH_CHECK(v.dim() == 5, "outlook_attention_custom: v must be 5D [B,NH,HWo,K2,HD]");

    const int64_t B   = attn.size(0);
    const int64_t NH  = attn.size(1);
    const int64_t HWo = attn.size(2);
    const int64_t K2o = attn.size(3);
    const int64_t K2i = attn.size(4);

    TORCH_CHECK(v.size(0) == B && v.size(1) == NH && v.size(2) == HWo, "outlook_attention_custom: v shape mismatch (B,NH,HWo)");
    TORCH_CHECK(v.size(3) == K2o, "outlook_attention_custom: v K2 mismatch");
    const int64_t HD = v.size(4);

    TORCH_CHECK(NH == 1, "outlook_attention_custom: only num_heads==1 supported");
    TORCH_CHECK(K2o == 9 && K2i == 9, "outlook_attention_custom: only kernel_size==3 supported (K2==9)");
    TORCH_CHECK(HWo == 49, "outlook_attention_custom: expects H=W=7, stride=1 so HWo==49");
    TORCH_CHECK(HD > 0, "outlook_attention_custom: head_dim must be > 0");

    const int64_t C = NH * HD;
    // Output is NCHW [B,C,7,7] as in the baseline fused tail.
    at::Tensor y = at::empty({B, C, 7, 7}, attn.options());
    EXEC_NPU_CMD(aclnnOutlookAttentionCustom, attn, v, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("outlook_attention_custom", &outlook_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("outlook_attention_custom", &outlook_attention_impl_npu,
          "OutlookAttention fused tail: fold(attn @ v) -> NCHW (NPU)");
}
