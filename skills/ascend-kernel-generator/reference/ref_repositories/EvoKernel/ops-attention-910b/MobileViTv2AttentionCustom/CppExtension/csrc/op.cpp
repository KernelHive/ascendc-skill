
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor mobile_vi_tv2_attention_impl_npu(
    const at::Tensor& i,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& w_o,
    const at::Tensor& b_o)
{
    TORCH_CHECK(i.device().type() == c10::DeviceType::PrivateUse1, "mobile_vi_tv2_attention_custom: i must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "mobile_vi_tv2_attention_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "mobile_vi_tv2_attention_custom: v must be on NPU");
    TORCH_CHECK(w_o.device().type() == c10::DeviceType::PrivateUse1, "mobile_vi_tv2_attention_custom: w_o must be on NPU");
    TORCH_CHECK(b_o.device().type() == c10::DeviceType::PrivateUse1, "mobile_vi_tv2_attention_custom: b_o must be on NPU");

    TORCH_CHECK(i.scalar_type() == at::kFloat, "mobile_vi_tv2_attention_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "mobile_vi_tv2_attention_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "mobile_vi_tv2_attention_custom: only float32 supported");
    TORCH_CHECK(w_o.scalar_type() == at::kFloat, "mobile_vi_tv2_attention_custom: only float32 supported");
    TORCH_CHECK(b_o.scalar_type() == at::kFloat, "mobile_vi_tv2_attention_custom: only float32 supported");

    TORCH_CHECK(i.is_contiguous(), "mobile_vi_tv2_attention_custom: i must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "mobile_vi_tv2_attention_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "mobile_vi_tv2_attention_custom: v must be contiguous");
    TORCH_CHECK(w_o.is_contiguous(), "mobile_vi_tv2_attention_custom: w_o must be contiguous");
    TORCH_CHECK(b_o.is_contiguous(), "mobile_vi_tv2_attention_custom: b_o must be contiguous");

    TORCH_CHECK(i.dim() == 3, "mobile_vi_tv2_attention_custom: i must be [bs,nq,1]");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "mobile_vi_tv2_attention_custom: k/v must be [bs,nq,d]");
    TORCH_CHECK(w_o.dim() == 2, "mobile_vi_tv2_attention_custom: w_o must be [d,d] (fc_o.weight)");
    TORCH_CHECK(b_o.dim() == 1, "mobile_vi_tv2_attention_custom: b_o must be [d] (fc_o.bias)");

    auto bs = i.size(0);
    auto nq = i.size(1);
    TORCH_CHECK(i.size(2) == 1, "mobile_vi_tv2_attention_custom: i.size(2) must be 1");

    TORCH_CHECK(k.size(0) == bs && k.size(1) == nq, "mobile_vi_tv2_attention_custom: k shape mismatch with i");
    TORCH_CHECK(v.size(0) == bs && v.size(1) == nq, "mobile_vi_tv2_attention_custom: v shape mismatch with i");

    auto d = k.size(2);
    TORCH_CHECK(d > 0, "mobile_vi_tv2_attention_custom: d must be > 0");
    TORCH_CHECK(v.size(2) == d, "mobile_vi_tv2_attention_custom: v.size(2) must equal k.size(2)");

    TORCH_CHECK(w_o.size(0) == d && w_o.size(1) == d, "mobile_vi_tv2_attention_custom: w_o must be [d,d] (fc_o.weight)");
    TORCH_CHECK(b_o.size(0) == d, "mobile_vi_tv2_attention_custom: b_o must be [d]");

    at::Tensor y = at::empty({bs, nq, d}, k.options());
    EXEC_NPU_CMD(aclnnMobileViTv2AttentionCustom, i, k, v, w_o, b_o, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mobile_vi_tv2_attention_custom", &mobile_vi_tv2_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mobile_vi_tv2_attention_custom", &mobile_vi_tv2_attention_impl_npu,
          "MobileViTv2 attention fused tail op (softmax+context+scale+fc_o) on NPU");
}
