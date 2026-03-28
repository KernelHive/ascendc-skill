
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor square_matrix_multiplication_custom_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D (N,N)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (N,N)");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square: A.size(0) == A.size(1)");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square: B.size(0) == B.size(1)");
    TORCH_CHECK(A.size(1) == B.size(0), "shape mismatch: A.size(1) must equal B.size(0)");
    TORCH_CHECK(A.size(0) == B.size(1), "square mismatch: output must be (N,N)");
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    at::Tensor C = at::empty({A.size(0), B.size(1)}, A.options());
    EXEC_NPU_CMD(aclnnSquareMatrixMultiplicationCustom, A, B, C);
    return C;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("square_matrix_multiplication_custom", &square_matrix_multiplication_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_matrix_multiplication_custom",
          &square_matrix_multiplication_custom_impl_npu,
          "C = A * B (square matmul, AscendC Matmul-backed, NPU)");
}
