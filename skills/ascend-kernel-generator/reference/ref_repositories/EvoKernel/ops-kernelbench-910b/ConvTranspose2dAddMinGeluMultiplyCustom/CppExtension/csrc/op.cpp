
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose2d_add_min_gelu_multiply_custom_impl_npu(const at::Tensor& x) {
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() >= 2, "x rank must be >= 2 (N,C,...)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnConvTranspose2dAddMinGeluMultiplyCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose2d_add_min_gelu_multiply_custom",
           &conv_transpose2d_add_min_gelu_multiply_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_add_min_gelu_multiply_custom",
          &conv_transpose2d_add_min_gelu_multiply_custom_impl_npu,
          "conv_transpose2d_add_min_gelu_multiply_custom(x) -> fused epilogue: (x+0.5)->min(.,0)->gelu(tanh-approx)->*2");
}
