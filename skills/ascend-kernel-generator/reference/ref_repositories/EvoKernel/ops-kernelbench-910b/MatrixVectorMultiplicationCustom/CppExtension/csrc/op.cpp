
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matrix_vector_multiplication_custom_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    // Strict contract: A (M,K), B (K,1) => C (M,1)
    TORCH_CHECK(A.dim() == 2, "A must be 2D (M,K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,1)");
    TORCH_CHECK(B.size(1) == 1, "B must have shape (K,1)");
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch: A.size(1) must equal B.size(0)");

    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");

    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");

    at::Tensor C = at::empty({A.size(0), 1}, A.options());
    EXEC_NPU_CMD(aclnnMatrixVectorMultiplicationCustom, A, B, C);
    return C;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matrix_vector_multiplication_custom", &matrix_vector_multiplication_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_vector_multiplication_custom",
          &matrix_vector_multiplication_custom_impl_npu,
          "Matrix-vector multiplication via Matmul (custom, NPU)");
}
