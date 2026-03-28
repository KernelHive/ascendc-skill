
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_with_small_k_dimension_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    // Compute: C = A @ B
    // A: (M,K), B: (K,N) => C: (M,N)
    TORCH_CHECK(A.dim() == 2, "A must be 2D (M,K) for this custom op");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,N) for this custom op");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch: A.size(1) must equal B.size(0)");
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (ND) for this custom op");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (ND) for this custom op");

    const int64_t M = A.size(0);
    const int64_t N = B.size(1);
    at::Tensor C = at::empty({M, N}, A.options());

    EXEC_NPU_CMD(aclnnMatmulWithSmallKDimensionCustom, A, B, C);
    return C;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_with_small_k_dimension_custom", &matmul_with_small_k_dimension_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_with_small_k_dimension_custom",
          &matmul_with_small_k_dimension_impl_npu,
          "Compute A @ B for A[M,K], B[K,N] (custom AscendC, small-K optimized, NPU)");
}
