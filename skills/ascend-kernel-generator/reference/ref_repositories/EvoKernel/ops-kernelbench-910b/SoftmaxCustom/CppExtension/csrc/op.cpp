
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor softmax_custom_impl_npu(const at::Tensor& x) {
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSoftmaxCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("softmax_custom", &softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_custom", &softmax_custom_impl_npu, "SoftmaxCustom (NPU)");
}
