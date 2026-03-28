
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static at::Tensor shuffle_attention_impl_npu(const at::Tensor& x0,
                                             const at::Tensor& x1,
                                             const at::Tensor& gate_c,
                                             const at::Tensor& s_norm,
                                             const at::Tensor& sweight,
                                             const at::Tensor& sbias,
                                             int64_t B,
                                             int64_t C)
{
    auto is_npu = [](const at::Tensor& t){ return t.device().type() == c10::DeviceType::PrivateUse1; };
    TORCH_CHECK(is_npu(x0) && is_npu(x1) && is_npu(gate_c) && is_npu(s_norm) && is_npu(sweight) && is_npu(sbias),
                "shuffle_attention_custom: all inputs must be on NPU");

    TORCH_CHECK(x0.scalar_type() == at::kFloat && x1.scalar_type() == at::kFloat &&
                gate_c.scalar_type() == at::kFloat && s_norm.scalar_type() == at::kFloat &&
                sweight.scalar_type() == at::kFloat && sbias.scalar_type() == at::kFloat,
                "shuffle_attention_custom: all inputs must be float32");

    TORCH_CHECK(x0.is_contiguous() && x1.is_contiguous() && gate_c.is_contiguous() &&
                s_norm.is_contiguous() && sweight.is_contiguous() && sbias.is_contiguous(),
                "shuffle_attention_custom: all inputs must be contiguous");

    TORCH_CHECK(x0.dim() == 4 && x1.dim() == 4 && s_norm.dim() == 4,
                "shuffle_attention_custom: x0/x1/s_norm must be 4D [BG,C2g,H,W]");
    TORCH_CHECK(gate_c.dim() == 4, "shuffle_attention_custom: gate_c must be 4D [BG,C2g,1,1]");
    TORCH_CHECK(sweight.dim() == 4 && sbias.dim() == 4,
                "shuffle_attention_custom: sweight/sbias must be 4D [1,C2g,1,1]");

    auto BG  = x0.size(0);
    auto C2g = x0.size(1);
    auto H   = x0.size(2);
    auto W   = x0.size(3);

    TORCH_CHECK(x1.sizes() == x0.sizes(), "shuffle_attention_custom: x1 shape must match x0");
    TORCH_CHECK(s_norm.sizes() == x0.sizes(), "shuffle_attention_custom: s_norm shape must match x0");

    TORCH_CHECK(gate_c.size(0) == BG && gate_c.size(1) == C2g && gate_c.size(2) == 1 && gate_c.size(3) == 1,
                "shuffle_attention_custom: gate_c must be [BG,C2g,1,1]");

    TORCH_CHECK(sweight.size(0) == 1 && sweight.size(1) == C2g && sweight.size(2) == 1 && sweight.size(3) == 1,
                "shuffle_attention_custom: sweight must be [1,C2g,1,1]");
    TORCH_CHECK(sbias.sizes() == sweight.sizes(), "shuffle_attention_custom: sbias must match sweight shape");

    TORCH_CHECK(B > 0 && C > 0, "shuffle_attention_custom: B and C must be positive");
    TORCH_CHECK(C == 2 * (BG / B) * C2g, "shuffle_attention_custom: C must equal 2*(BG/B)*C2g");
    TORCH_CHECK(BG % B == 0, "shuffle_attention_custom: BG must be divisible by B");

    at::Tensor y = at::empty({B, C, H, W}, x0.options());
    EXEC_NPU_CMD(aclnnShuffleAttentionCustom, x0, x1, gate_c, s_norm, sweight, sbias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("shuffle_attention_custom", &shuffle_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shuffle_attention_custom", &shuffle_attention_impl_npu,
          "ShuffleAttention fused tail + correct channel_shuffle(groups=2) over [B,C,H,W] (NPU, float32)");
}
