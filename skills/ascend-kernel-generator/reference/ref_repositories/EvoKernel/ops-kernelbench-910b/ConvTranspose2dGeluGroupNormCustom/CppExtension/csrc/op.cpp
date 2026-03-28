
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose2d_gelu_group_norm_custom_impl_npu(const at::Tensor& x,
                                                            const at::Tensor& gamma,
                                                            const at::Tensor& beta) {
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(gamma.is_privateuseone(), "gamma must be on NPU");
    TORCH_CHECK(beta.is_privateuseone(), "beta must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
    TORCH_CHECK(x.dim() >= 2, "x rank must be >= 2 (N,C,...)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (C)");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D (C)");
    TORCH_CHECK(gamma.numel() == x.size(1), "gamma must have C elements");
    TORCH_CHECK(beta.numel() == x.size(1), "beta must have C elements");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnConvTranspose2dGeluGroupNormCustom, x, gamma, beta, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose2d_gelu_group_norm_custom",
           &conv_transpose2d_gelu_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_gelu_group_norm_custom",
          &conv_transpose2d_gelu_group_norm_custom_impl_npu,
          "conv_transpose2d_gelu_group_norm_custom(x, gamma, beta) -> fused GELU(tanh-approx) + GroupNorm (baked num_groups=8, eps=1e-5)");
}
