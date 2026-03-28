
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor elu_impl_npu(const at::Tensor& self, double alpha) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnEluCustom, self, alpha, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("elu_custom", &elu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elu_custom", &elu_impl_npu, "ELU activation (AscendC)");
}
