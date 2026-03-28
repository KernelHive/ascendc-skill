
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor sum_reduction_over_a_dimension_custom_impl_npu(const at::Tensor& x, int64_t dim)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "expects 3D input [B,N,S]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t d = dim;
    if (d < 0) d += x.dim();
    TORCH_CHECK(d == 1, "only dim==1 (or -2) supported");

    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 4096 && x.size(2) == 4095,
                "only supports shape [128,4096,4095]");

    at::Tensor y = at::empty({x.size(0), 1, x.size(2)}, x.options());
    EXEC_NPU_CMD(aclnnSumReductionOverADimensionCustom, x, d, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("sum_reduction_over_a_dimension_custom(Tensor x, int dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduction_over_a_dimension_custom",
          &sum_reduction_over_a_dimension_custom_impl_npu,
          "sum reduction over a dimension (keepdim) on NPU");
}
