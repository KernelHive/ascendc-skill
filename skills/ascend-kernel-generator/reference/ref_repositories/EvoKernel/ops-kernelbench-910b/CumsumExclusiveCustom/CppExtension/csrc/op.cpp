
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor cumsum_exclusive_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.defined(), "cumsum_exclusive_custom: input must be defined");
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "cumsum_exclusive_custom: expected NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "cumsum_exclusive_custom: only float32 is supported");
    TORCH_CHECK(x.dim() == 2, "cumsum_exclusive_custom: specialized for 2D input (rows, cols)");
    TORCH_CHECK(x.is_contiguous(), "cumsum_exclusive_custom: input must be contiguous");

    auto y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnCumsumExclusiveCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cumsum_exclusive_custom", &cumsum_exclusive_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_exclusive_custom", &cumsum_exclusive_impl_npu, "exclusive cumsum along last dim (NPU)");
}
