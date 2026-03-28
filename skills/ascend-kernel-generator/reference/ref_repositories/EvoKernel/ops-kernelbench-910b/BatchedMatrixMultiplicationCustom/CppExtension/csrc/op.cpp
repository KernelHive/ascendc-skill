
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor batched_matrix_multiplication_custom_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D (batch, M, K) for batched_matrix_multiplication_custom");
    TORCH_CHECK(B.dim() == 3, "B must be 3D (batch, K, N) for batched_matrix_multiplication_custom");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch mismatch: A.size(0) must equal B.size(0)");
    TORCH_CHECK(A.size(2) == B.size(1), "K mismatch: A.size(2) must equal B.size(1)");

    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");

    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (ND) for this custom op");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (ND) for this custom op");

    const int64_t batch = A.size(0);
    const int64_t M = A.size(1);
    const int64_t N = B.size(2);

    at::Tensor C = at::empty({batch, M, N}, A.options());

    EXEC_NPU_CMD(aclnnBatchedMatrixMultiplicationCustom, A, B, C);
    return C;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("batched_matrix_multiplication_custom", &batched_matrix_multiplication_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matrix_multiplication_custom",
          &batched_matrix_multiplication_custom_impl_npu,
          "Compute C = bmm(A, B) for A[batch,M,K], B[batch,K,N] (custom AscendC, NPU)");
}
