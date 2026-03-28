
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

void log_softmax_custom_out_impl_npu(const at::Tensor& x, at::Tensor& out) {
    EXEC_NPU_CMD(aclnnLogSoftmaxCustom, x, out);
}

at::Tensor log_softmax_custom_impl_npu(const at::Tensor& x) {
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnLogSoftmaxCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("log_softmax_custom", &log_softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("log_softmax_custom", &log_softmax_custom_impl_npu, "LogSoftmaxCustom (NPU)");
    m.def("log_softmax_custom_out", &log_softmax_custom_out_impl_npu, "LogSoftmaxCustom out= (NPU)");
}
