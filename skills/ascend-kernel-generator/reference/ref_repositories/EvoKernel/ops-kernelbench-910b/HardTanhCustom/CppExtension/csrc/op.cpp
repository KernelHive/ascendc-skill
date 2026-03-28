
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor& hard_tanh_custom_out_impl_npu(const at::Tensor& self, at::Tensor& out) {
    TORCH_CHECK(self.device().type() == out.device().type(), "self/out device mismatch");
    TORCH_CHECK(self.scalar_type() == out.scalar_type(), "self/out dtype mismatch");
    TORCH_CHECK(self.numel() == out.numel(), "self/out numel mismatch");
    EXEC_NPU_CMD(aclnnHardTanhCustom, self, out);
    return out;
}

at::Tensor hard_tanh_custom_impl_npu(const at::Tensor& self) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnHardTanhCustom, self, result);
    return result;
}

TORCH_LIBRARY(myops, m) {
    m.def("hard_tanh_custom(Tensor x) -> Tensor");
    m.def("hard_tanh_custom_out(Tensor x, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("hard_tanh_custom", &hard_tanh_custom_impl_npu);
    m.impl("hard_tanh_custom_out", &hard_tanh_custom_out_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hard_tanh_custom", &hard_tanh_custom_impl_npu,
          "hard_tanh_custom(x) -> clamp(x, -1, 1) (NPU)");
    m.def("hard_tanh_custom_out", &hard_tanh_custom_out_impl_npu,
          "hard_tanh_custom_out(x, out) -> out (NPU)");
}
