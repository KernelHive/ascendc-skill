
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& q_phi,
                         const at::Tensor& k_phi,
                         const at::Tensor& v)
{
    TORCH_CHECK(q_phi.device().type() == c10::DeviceType::PrivateUse1, "performer_attention_custom: q_phi must be on NPU");
    TORCH_CHECK(k_phi.device().type() == c10::DeviceType::PrivateUse1, "performer_attention_custom: k_phi must be on NPU");
    TORCH_CHECK(v.device().type()     == c10::DeviceType::PrivateUse1, "performer_attention_custom: v must be on NPU");

    TORCH_CHECK(q_phi.scalar_type() == at::kFloat, "performer_attention_custom: q_phi must be float32");
    TORCH_CHECK(k_phi.scalar_type() == at::kFloat, "performer_attention_custom: k_phi must be float32");
    TORCH_CHECK(v.scalar_type()     == at::kFloat, "performer_attention_custom: v must be float32");

    TORCH_CHECK(q_phi.is_contiguous(), "performer_attention_custom: q_phi must be contiguous");
    TORCH_CHECK(k_phi.is_contiguous(), "performer_attention_custom: k_phi must be contiguous");
    TORCH_CHECK(v.is_contiguous(),     "performer_attention_custom: v must be contiguous");

    TORCH_CHECK(q_phi.dim() == 4, "performer_attention_custom: q_phi must be 4D [B,H,S,F]");
    TORCH_CHECK(k_phi.dim() == 4, "performer_attention_custom: k_phi must be 4D [B,H,S,F]");
    TORCH_CHECK(v.dim() == 4,     "performer_attention_custom: v must be 4D [B,H,S,D]");

    const auto B = q_phi.size(0);
    const auto H = q_phi.size(1);
    const auto S = q_phi.size(2);
    const auto F = q_phi.size(3);

    TORCH_CHECK(B > 0 && H > 0 && S > 0 && F > 0, "performer_attention_custom: empty dims not supported");

    TORCH_CHECK(k_phi.size(0) == B && k_phi.size(1) == H && k_phi.size(2) == S && k_phi.size(3) == F,
                "performer_attention_custom: k_phi must match q_phi [B,H,S,F]");

    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == S,
                "performer_attention_custom: v must match [B,H,S,*]");
    const auto D = v.size(3);
    TORCH_CHECK(D > 0, "performer_attention_custom: D must be > 0");

    // Must match kernel/host caps.
    TORCH_CHECK(S <= 4096, "performer_attention_custom: S > 4096 not supported");
    TORCH_CHECK(F <= 256,  "performer_attention_custom: F > 256 not supported");
    TORCH_CHECK(D <= 128,  "performer_attention_custom: D > 128 not supported");
}

at::Tensor performer_attention_custom_impl_npu(const at::Tensor& q_phi,
                                               const at::Tensor& k_phi,
                                               const at::Tensor& v)
{
    check_inputs(q_phi, k_phi, v);
    auto y = at::empty({q_phi.size(0), q_phi.size(1), q_phi.size(2), v.size(3)}, q_phi.options());
    EXEC_NPU_CMD(aclnnPerformerAttentionCustom, q_phi, k_phi, v, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("performer_attention_custom", &performer_attention_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("performer_attention_custom",
          &performer_attention_custom_impl_npu,
          "PerformerAttentionCustom on NPU (q_phi/k_phi:[B,H,S,F], v:[B,H,S,D]) -> y:[B,H,S,D]");
}
