
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor cross_entropy_loss_custom_impl_npu(const at::Tensor& predict, const at::Tensor& target) {
    at::Tensor result = at::empty({}, predict.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnCrossEntropyLossCustom, predict, target, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cross_entropy_loss_custom", &cross_entropy_loss_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_entropy_loss_custom", &cross_entropy_loss_custom_impl_npu, "cross entropy loss (custom, NPU)");
}
