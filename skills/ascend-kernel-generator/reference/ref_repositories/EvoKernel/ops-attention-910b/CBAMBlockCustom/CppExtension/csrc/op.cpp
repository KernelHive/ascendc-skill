
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor cbam_block_impl_npu(const at::Tensor& x,
                              const at::Tensor& ca,
                              const at::Tensor& sa)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "cbam_block_custom: x must be on NPU");
    TORCH_CHECK(ca.device().type() == c10::DeviceType::PrivateUse1, "cbam_block_custom: ca must be on NPU");
    TORCH_CHECK(sa.device().type() == c10::DeviceType::PrivateUse1, "cbam_block_custom: sa must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "cbam_block_custom: only float32 supported");
    TORCH_CHECK(ca.scalar_type() == at::kFloat, "cbam_block_custom: ca must be float32");
    TORCH_CHECK(sa.scalar_type() == at::kFloat, "cbam_block_custom: sa must be float32");

    TORCH_CHECK(x.dim() == 4, "cbam_block_custom: x must be [B,C,H,W]");
    TORCH_CHECK(ca.dim() == 4, "cbam_block_custom: ca must be [B,C,1,1]");
    TORCH_CHECK(sa.dim() == 4, "cbam_block_custom: sa must be [B,1,H,W]");

    TORCH_CHECK(x.is_contiguous(), "cbam_block_custom: x must be contiguous (NCHW)");
    TORCH_CHECK(ca.is_contiguous(), "cbam_block_custom: ca must be contiguous");
    TORCH_CHECK(sa.is_contiguous(), "cbam_block_custom: sa must be contiguous");

    const auto B = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0, "cbam_block_custom: invalid x shape");

    TORCH_CHECK(ca.size(0) == B && ca.size(1) == C && ca.size(2) == 1 && ca.size(3) == 1,
                "cbam_block_custom: ca must be [B,C,1,1] matching x");
    TORCH_CHECK(sa.size(0) == B && sa.size(1) == 1 && sa.size(2) == H && sa.size(3) == W,
                "cbam_block_custom: sa must be [B,1,H,W] matching x");

    at::Tensor y = at::empty_like(x);
    // Keep naming consistent with op type: CBAMBlockCustom -> aclnnCBAMBlockCustom
    EXEC_NPU_CMD(aclnnCBAMBlockCustom, x, ca, sa, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cbam_block_custom", &cbam_block_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cbam_block_custom", &cbam_block_impl_npu,
          "CBAM fused tail on NPU: y = x * ca * sa + x (float32, NCHW)");
}
