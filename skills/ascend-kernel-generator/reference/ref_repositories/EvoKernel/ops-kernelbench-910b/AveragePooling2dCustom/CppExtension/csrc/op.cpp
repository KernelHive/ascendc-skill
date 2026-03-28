
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor average_pooling2d_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");

    constexpr int64_t KH = 11;
    constexpr int64_t KW = 11;
    constexpr int64_t SH = 11;
    constexpr int64_t SW = 11;
    constexpr int64_t PH = 0;
    constexpr int64_t PW = 0;

    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t Ho = (H + 2 * PH - KH) / SH + 1;
    const int64_t Wo = (W + 2 * PW - KW) / SW + 1;
    TORCH_CHECK(Ho > 0 && Wo > 0, "invalid output spatial size");

    at::Tensor y = at::empty({x.size(0), x.size(1), Ho, Wo}, x.options());
    EXEC_NPU_CMD(aclnnAveragePooling2dCustom, x, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("average_pooling2d_custom(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling2d_custom", &average_pooling2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("average_pooling2d_custom", &average_pooling2d_custom_impl_npu, "average_pooling2d_custom (NPU)");
}
