
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor max_pooling2d_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");

    constexpr int64_t K = 4;
    constexpr int64_t S = 1;
    constexpr int64_t P = 1;
    constexpr int64_t D = 1;

    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t Ho = (H + 2 * P - (K - 1) * D - 1) / S + 1;
    const int64_t Wo = (W + 2 * P - (K - 1) * D - 1) / S + 1;
    TORCH_CHECK(Ho > 0 && Wo > 0, "invalid output spatial size");

    TORCH_CHECK(K == 4 && S == 1 && P == 1 && D == 1,
                "max_pooling2d_custom compiled for k=4,s=1,p=1,d=1 only");

    at::Tensor y = at::empty({x.size(0), x.size(1), Ho, Wo}, x.options());
    EXEC_NPU_CMD(aclnnMaxPooling2dCustom, x, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("max_pooling2d_custom(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("max_pooling2d_custom", &max_pooling2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pooling2d_custom", &max_pooling2d_custom_impl_npu, "max_pooling2d_custom (NPU)");
}
