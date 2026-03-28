
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor cumsum_reverse_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.defined(), "cumsum_reverse_custom: input must be defined");
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "cumsum_reverse_custom: expected NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "cumsum_reverse_custom: only float32 is supported");
    TORCH_CHECK(x.dim() == 2, "cumsum_reverse_custom: specialized for 2D input (rows, cols)");
    TORCH_CHECK(x.is_contiguous(), "cumsum_reverse_custom: input must be contiguous");

    auto y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnCumsumReverseCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cumsum_reverse_custom", &cumsum_reverse_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_reverse_custom", &cumsum_reverse_impl_npu, "reverse cumsum along last dim (NPU)");
}
