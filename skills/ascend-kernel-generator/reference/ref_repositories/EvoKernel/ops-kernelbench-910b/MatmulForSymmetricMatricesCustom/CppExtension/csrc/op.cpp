
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_for_symmetric_matrices_custom_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D (N,N)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (N,N)");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (ND)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (ND)");

    at::Tensor C = at::empty({A.size(0), B.size(1)}, A.options());
    EXEC_NPU_CMD(aclnnMatmulForSymmetricMatricesCustom, A, B, C);
    return C;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_for_symmetric_matrices_custom", &matmul_for_symmetric_matrices_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_for_symmetric_matrices_custom",
          &matmul_for_symmetric_matrices_custom_impl_npu,
          "matmul for symmetric matrices (exact A*B, NPU)");
}
