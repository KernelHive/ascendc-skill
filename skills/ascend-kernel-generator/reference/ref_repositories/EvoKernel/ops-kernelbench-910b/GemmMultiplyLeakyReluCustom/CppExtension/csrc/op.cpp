
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_multiply_leaky_relu_custom_impl_npu(const at::Tensor& x,
                                                   const at::Tensor& w,
                                                   const at::Tensor& b,
                                                   const at::Tensor& multiplier,
                                                   const at::Tensor& negative_slope)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "w must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "b must be on NPU");
    TORCH_CHECK(multiplier.device().type() == c10::DeviceType::PrivateUse1, "multiplier must be on NPU");
    TORCH_CHECK(negative_slope.device().type() == c10::DeviceType::PrivateUse1, "negative_slope must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(multiplier.scalar_type() == at::kFloat, "multiplier must be float32");
    TORCH_CHECK(negative_slope.scalar_type() == at::kFloat, "negative_slope must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [N]");

    TORCH_CHECK(multiplier.numel() == 1, "multiplier must be a scalar tensor (numel==1)");
    TORCH_CHECK(negative_slope.numel() == 1, "negative_slope must be a scalar tensor (numel==1)");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(multiplier.is_contiguous(), "multiplier must be contiguous");
    TORCH_CHECK(negative_slope.is_contiguous(), "negative_slope must be contiguous");

    // Fixed specialized contract enforced for kernel/tiling simplicity.
    TORCH_CHECK(x.size(0) == 1024 && x.size(1) == 8192,
                "x shape must be [1024,8192] for gemm_multiply_leaky_relu_custom");
    TORCH_CHECK(w.size(0) == 8192 && w.size(1) == 8192,
                "w shape must be [8192,8192] (out,in) for gemm_multiply_leaky_relu_custom");
    TORCH_CHECK(b.size(0) == 8192,
                "b shape must be [8192] for gemm_multiply_leaky_relu_custom");

    auto y = at::empty({1024, 8192}, x.options());
    EXEC_NPU_CMD(aclnnGemmMultiplyLeakyReluCustom, x, w, b, multiplier, negative_slope, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_multiply_leaky_relu_custom", &gemm_multiply_leaky_relu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_multiply_leaky_relu_custom", &gemm_multiply_leaky_relu_custom_impl_npu,
          "gemm_multiply_leaky_relu_custom (NPU)");
}
