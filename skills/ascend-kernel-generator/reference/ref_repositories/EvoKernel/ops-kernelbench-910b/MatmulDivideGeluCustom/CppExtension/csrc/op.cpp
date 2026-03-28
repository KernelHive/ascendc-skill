
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_divide_gelu_custom_impl_npu(const at::Tensor& x,
                                             const at::Tensor& w,
                                             const at::Tensor& b,
                                             const at::Tensor& divisor)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "w must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "b must be on NPU");
    TORCH_CHECK(divisor.device().type() == c10::DeviceType::PrivateUse1, "divisor must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(divisor.scalar_type() == at::kFloat, "divisor must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [N]");
    TORCH_CHECK((divisor.dim() == 0) || (divisor.dim() == 1 && divisor.numel() == 1),
                "divisor must be scalar (0D or [1])");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(divisor.is_contiguous(), "divisor must be contiguous");

    TORCH_CHECK(x.size(0) == 1024 && x.size(1) == 8192,
                "x shape must be [1024,8192] for matmul_divide_gelu_custom");
    TORCH_CHECK(w.size(0) == 8192 && w.size(1) == 8192,
                "w shape must be [8192,8192] (out,in) for matmul_divide_gelu_custom");
    TORCH_CHECK(b.size(0) == 8192, "b shape must be [8192] for matmul_divide_gelu_custom");

    auto y = at::empty({1024, 8192}, x.options());
    EXEC_NPU_CMD(aclnnMatmulDivideGeluCustom, x, w, b, divisor, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_divide_gelu_custom", &matmul_divide_gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_divide_gelu_custom", &matmul_divide_gelu_custom_impl_npu,
          "matmul_divide_gelu_custom (NPU)");
}
