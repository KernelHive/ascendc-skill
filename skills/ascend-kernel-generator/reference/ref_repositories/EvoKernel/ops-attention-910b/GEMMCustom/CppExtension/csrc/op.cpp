
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void gemm_custom_check(const at::Tensor& x, const at::Tensor& weight_t) {
    TORCH_CHECK(x.defined() && weight_t.defined(), "gemm_custom: inputs must be defined");
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "gemm_custom: x must be on NPU");
    TORCH_CHECK(weight_t.device().type() == at::kPrivateUse1, "gemm_custom: weight_t must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "gemm_custom: x must be bfloat16");
    TORCH_CHECK(weight_t.scalar_type() == at::kBFloat16, "gemm_custom: weight_t must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "gemm_custom: x must be 2D (M,K), got dim=", x.dim());
    TORCH_CHECK(weight_t.dim() == 2, "gemm_custom: weight_t must be 2D (K,N), got dim=", weight_t.dim());
    TORCH_CHECK(x.size(1) == weight_t.size(0),
                "gemm_custom: K mismatch: x.size(1)=", x.size(1), " weight_t.size(0)=", weight_t.size(0));
    TORCH_CHECK(x.is_contiguous(), "gemm_custom: x must be contiguous");
    TORCH_CHECK(weight_t.is_contiguous(), "gemm_custom: weight_t must be contiguous (call .contiguous() after transpose)");
}

at::Tensor gemm_custom_impl_npu(const at::Tensor& x, const at::Tensor& weight_t) {
    gemm_custom_check(x, weight_t);
    const int64_t M = x.size(0);
    const int64_t N = weight_t.size(1);
    at::Tensor y = at::empty({M, N}, x.options());
    EXEC_NPU_CMD(aclnnGEMMCustom, x, weight_t, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_custom", &gemm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_custom", &gemm_custom_impl_npu, "GEMMCustom: y = x @ weight_t (x: MxK, weight_t: KxN)");
}
