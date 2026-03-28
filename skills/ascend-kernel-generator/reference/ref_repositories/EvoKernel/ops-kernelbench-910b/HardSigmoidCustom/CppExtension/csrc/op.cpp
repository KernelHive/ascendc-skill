
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor& hard_sigmoid_custom_out_impl_npu(const at::Tensor& self, at::Tensor& out) {
    TORCH_CHECK(self.device().type() == out.device().type(), "self/out device mismatch");
    TORCH_CHECK(self.scalar_type() == out.scalar_type(), "self/out dtype mismatch");
    TORCH_CHECK(self.numel() == out.numel(), "self/out numel mismatch");
    EXEC_NPU_CMD(aclnnHardSigmoidCustom, self, out);
    return out;
}

at::Tensor hard_sigmoid_custom_impl_npu(const at::Tensor& self) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnHardSigmoidCustom, self, result);
    return result;
}

TORCH_LIBRARY(myops, m) {
    m.def("hard_sigmoid_custom(Tensor x) -> Tensor");
    m.def("hard_sigmoid_custom_out(Tensor x, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("hard_sigmoid_custom", &hard_sigmoid_custom_impl_npu);
    m.impl("hard_sigmoid_custom_out", &hard_sigmoid_custom_out_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hard_sigmoid_custom", &hard_sigmoid_custom_impl_npu,
          "hard_sigmoid_custom(x) -> clamp(x + 3, 0, 6) / 6 (NPU)");
    m.def("hard_sigmoid_custom_out", &hard_sigmoid_custom_out_impl_npu,
          "hard_sigmoid_custom_out(x, out) -> out (NPU, avoids extra allocation)");
}
