
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_add_relu_custom_impl_npu(const at::Tensor& x,
                                        const at::Tensor& w,
                                        const at::Tensor& bias)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "w must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [N]");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.size(0) == 1024 && x.size(1) == 8192,
                "x shape must be [1024,8192] for gemm_add_relu_custom");
    TORCH_CHECK(w.size(0) == 8192 && w.size(1) == 8192,
                "w shape must be [8192,8192] (out,in) for gemm_add_relu_custom");
    TORCH_CHECK(bias.size(0) == 8192,
                "bias shape must be [8192] for gemm_add_relu_custom");

    auto y = at::empty({1024, 8192}, x.options());
    EXEC_NPU_CMD(aclnnGemmAddReluCustom, x, w, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_add_relu_custom", &gemm_add_relu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_add_relu_custom", &gemm_add_relu_custom_impl_npu, "gemm_add_relu_custom (NPU)");
}
