
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor frobenius_norm_custom_impl_npu(const at::Tensor& x) {
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnFrobeniusNormCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("frobenius_norm_custom", &frobenius_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frobenius_norm_custom", &frobenius_norm_custom_impl_npu,
          "frobenius_norm_custom(x) -> x / ||x||_F (global Frobenius norm, float32)");
}
