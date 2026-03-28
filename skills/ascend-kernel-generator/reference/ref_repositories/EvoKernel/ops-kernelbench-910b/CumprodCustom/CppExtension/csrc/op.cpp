
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor cumprod_impl_npu(const at::Tensor& x)
{
    auto y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnCumprodCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cumprod_custom", &cumprod_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumprod_custom", &cumprod_impl_npu, "cumprod custom (NPU)");
}
