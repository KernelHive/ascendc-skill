
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor leaky_relu_impl_npu(const at::Tensor& self, double negative_slope) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnLeakyReluCustom, self, negative_slope, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("leaky_relu_custom", &leaky_relu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_custom", &leaky_relu_impl_npu, "LeakyReLU activation (LeakyReluCustom on NPU)");
}
