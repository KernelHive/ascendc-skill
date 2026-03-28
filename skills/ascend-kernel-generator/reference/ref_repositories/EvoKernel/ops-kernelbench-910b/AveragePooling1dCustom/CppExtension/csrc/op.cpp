
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor average_pooling1d_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N,C,L)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    constexpr int64_t K = 8;
    constexpr int64_t S = 1;
    constexpr int64_t P = 4;

    const int64_t L = x.size(2);
    const int64_t Lout = (L + 2 * P - K) / S + 1;
    TORCH_CHECK(Lout > 0, "invalid output length");

    at::Tensor y = at::empty({x.size(0), x.size(1), Lout}, x.options());
    EXEC_NPU_CMD(aclnnAveragePooling1dCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling1d_custom", &average_pooling1d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("average_pooling1d_custom", &average_pooling1d_custom_impl_npu, "average_pooling1d_custom (NPU)");
}
