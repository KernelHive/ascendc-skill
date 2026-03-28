
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <cmath>
#include <cstdint>

static inline int64_t I64SqrtRound(int64_t n) {
    return static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(n))));
}

at::Tensor global_filter_impl_npu(const at::Tensor& x,
                                 const at::Tensor& complex_weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "global_filter_custom: x must be on NPU");
    TORCH_CHECK(complex_weight.device().type() == c10::DeviceType::PrivateUse1, "global_filter_custom: complex_weight must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "global_filter_custom: only float32 supported for x");
    TORCH_CHECK(complex_weight.scalar_type() == at::kFloat, "global_filter_custom: only float32 supported for complex_weight");

    TORCH_CHECK(x.dim() == 3, "global_filter_custom: x must be [B,N,C]");
    TORCH_CHECK(complex_weight.dim() == 4, "global_filter_custom: complex_weight must be [H,FW,C,2]");

    TORCH_CHECK(x.is_contiguous(), "global_filter_custom: x must be contiguous");
    TORCH_CHECK(complex_weight.is_contiguous(), "global_filter_custom: complex_weight must be contiguous");

    const auto N = x.size(1);
    const auto C = x.size(2);

    const auto H = I64SqrtRound(N);
    TORCH_CHECK(H * H == N, "global_filter_custom: requires N to be a perfect square (spatial_size=None path)");
    const auto FW = H / 2 + 1;

    TORCH_CHECK(C == 512, "global_filter_custom: specialized for C=512");
    TORCH_CHECK(H == 7 && FW == 4, "global_filter_custom: specialized for N=49 (7x7) and FW=4");
    TORCH_CHECK(N == 49, "global_filter_custom: specialized for N=49");

    TORCH_CHECK(complex_weight.size(0) == H, "global_filter_custom: weight.size(0) must equal H");
    TORCH_CHECK(complex_weight.size(1) == FW, "global_filter_custom: weight.size(1) must equal FW");
    TORCH_CHECK(complex_weight.size(2) == C, "global_filter_custom: weight.size(2) must equal C");
    TORCH_CHECK(complex_weight.size(3) == 2, "global_filter_custom: weight last dim must be 2 (re,im)");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnGlobalFilterCustom, x, complex_weight, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("global_filter_custom(Tensor x, Tensor complex_weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("global_filter_custom", &global_filter_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("global_filter_custom", &global_filter_impl_npu,
          "GlobalFilter fused rfft2/mul/irfft2 (ortho) specialized for 7x7,C=512 on NPU");
}
