
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor pam_impl_npu(const at::Tensor& b,
                        const at::Tensor& c,
                        const at::Tensor& d)
{
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "pam_custom: b must be on NPU");
    TORCH_CHECK(c.device().type() == c10::DeviceType::PrivateUse1, "pam_custom: c must be on NPU");
    TORCH_CHECK(d.device().type() == c10::DeviceType::PrivateUse1, "pam_custom: d must be on NPU");

    TORCH_CHECK(b.scalar_type() == at::kFloat, "pam_custom: b must be float32");
    TORCH_CHECK(c.scalar_type() == at::kFloat, "pam_custom: c must be float32");
    TORCH_CHECK(d.scalar_type() == at::kFloat, "pam_custom: d must be float32");

    TORCH_CHECK(b.is_contiguous(), "pam_custom: b must be contiguous [N,S,C]");
    TORCH_CHECK(c.is_contiguous(), "pam_custom: c must be contiguous [N,C,S]");
    TORCH_CHECK(d.is_contiguous(), "pam_custom: d must be contiguous [N,S,C]");

    TORCH_CHECK(b.dim() == 3 && c.dim() == 3 && d.dim() == 3, "pam_custom: inputs must be 3D");
    const auto N = b.size(0);
    const auto S = b.size(1);
    const auto C = b.size(2);

    TORCH_CHECK(c.size(0) == N && c.size(1) == C && c.size(2) == S, "pam_custom: c must be [N,C,S]");
    TORCH_CHECK(d.size(0) == N && d.size(1) == S && d.size(2) == C, "pam_custom: d must be [N,S,C]");

    TORCH_CHECK(S > 0 && S <= 64, "pam_custom: only supports S in (0,64] (e.g., 7*7=49)");
    TORCH_CHECK((C % 16) == 0, "pam_custom: C must be multiple of 16 (e.g., 512)");

    at::Tensor y = at::empty({N, C, S}, b.options());
    EXEC_NPU_CMD(aclnnPAMCustom, b, c, d, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("pam_custom", &pam_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pam_custom", &pam_impl_npu,
          "PAM fused core on NPU: y_cs=[N,C,S] from B[N,S,C],C[N,C,S],D[N,S,C] (float32, S<=64)");
}
