
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor argmax_over_a_dimension_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, R, I]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (ND)");

    // Specialized fixed contract for this kernel.
    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 4096 && x.size(2) == 4095,
                "x shape must be [128,4096,4095] for argmax_over_a_dimension_custom");

    auto y = at::empty({128, 4095}, x.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnArgmaxOverADimensionCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("argmax_over_a_dimension_custom", &argmax_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("argmax_over_a_dimension_custom", &argmax_over_a_dimension_custom_impl_npu,
          "argmax_over_a_dimension_custom (NPU)");
}
