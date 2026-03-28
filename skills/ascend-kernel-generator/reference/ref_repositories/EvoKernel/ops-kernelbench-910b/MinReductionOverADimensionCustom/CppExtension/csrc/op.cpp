
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor min_reduction_over_a_dimension_custom_impl_npu(const at::Tensor& x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "min_reduction_over_a_dimension_custom: expected NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "min_reduction_over_a_dimension_custom: expected float32");
    TORCH_CHECK(x.dim() == 3,
                "min_reduction_over_a_dimension_custom: expected 3D tensor");
    TORCH_CHECK(x.is_contiguous(),
                "min_reduction_over_a_dimension_custom: expected contiguous tensor");
    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 4096 && x.size(2) == 4095,
                "min_reduction_over_a_dimension_custom: specialized to [128,4096,4095] and dim==1");

    auto y = at::empty({x.size(0), x.size(2)}, x.options());
    EXEC_NPU_CMD(aclnnMinReductionOverADimensionCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("min_reduction_over_a_dimension_custom", &min_reduction_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_reduction_over_a_dimension_custom",
          &min_reduction_over_a_dimension_custom_impl_npu,
          "Min reduction over dim=1 for [128,4096,4095] (custom, NPU)");
}
