
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_size_1d(int64_t in, int64_t k, int64_t s, int64_t p)
{
    return (in + 2 * p - k) / s + 1;
}

at::Tensor average_pooling3d_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().is_privateuseone(), "average_pooling3d_custom: x must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "average_pooling3d_custom: only float32 supported");
    TORCH_CHECK(x.dim() == 5, "average_pooling3d_custom: expected 5D NCDHW input");
    TORCH_CHECK(x.is_contiguous(), "average_pooling3d_custom: x must be contiguous");

    constexpr int64_t K = 3;
    constexpr int64_t S = 2;
    constexpr int64_t P = 1;

    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    const int64_t Od = out_size_1d(D, K, S, P);
    const int64_t Oh = out_size_1d(H, K, S, P);
    const int64_t Ow = out_size_1d(W, K, S, P);
    TORCH_CHECK(Od > 0 && Oh > 0 && Ow > 0, "average_pooling3d_custom: invalid output size");

    at::Tensor y = at::empty({x.size(0), x.size(1), Od, Oh, Ow}, x.options());
    EXEC_NPU_CMD(aclnnAveragePooling3dCustom, x, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("average_pooling3d_custom(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling3d_custom", &average_pooling3d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("average_pooling3d_custom", &average_pooling3d_custom_impl_npu, "average_pooling3d_custom (NPU)");
}
