
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_dim_floor(int64_t in, int64_t k, int64_t s, int64_t p, int64_t d) {
    return (in + 2 * p - (k - 1) * d - 1) / s + 1;
}

at::Tensor max_pooling3d_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    constexpr int64_t K = 3;
    constexpr int64_t S = 2;
    constexpr int64_t P = 1;
    constexpr int64_t DIL = 1;
    constexpr bool CEIL_MODE = false;
    constexpr bool RETURN_INDICES = false;
    (void)CEIL_MODE;
    (void)RETURN_INDICES;

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    TORCH_CHECK(DIL == 1, "dilation must be 1");
    TORCH_CHECK(K == 3 && S == 2 && P == 1, "operator specialized to kernel=3,stride=2,pad=1");

    const int64_t Do = out_dim_floor(D, K, S, P, DIL);
    const int64_t Ho = out_dim_floor(H, K, S, P, DIL);
    const int64_t Wo = out_dim_floor(W, K, S, P, DIL);
    TORCH_CHECK(Do > 0 && Ho > 0 && Wo > 0, "invalid output shape");

    at::Tensor y = at::empty({N, C, Do, Ho, Wo}, x.options());
    EXEC_NPU_CMD(aclnnMaxPooling3dCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("max_pooling3d_custom", &max_pooling3d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pooling3d_custom", &max_pooling3d_custom_impl_npu, "max_pooling3d_custom (NPU)");
}
