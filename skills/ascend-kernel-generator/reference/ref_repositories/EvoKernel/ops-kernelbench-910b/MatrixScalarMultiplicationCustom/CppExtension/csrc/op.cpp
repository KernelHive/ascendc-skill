
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline at::Tensor MakeScalarTensorOnNpuLike(const at::Tensor& like, double s_double)
{
    float s = static_cast<float>(s_double);
    at::Tensor s_cpu = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    s_cpu.fill_(s);
    // Force a real copy onto NPU to avoid any aliasing/constant-folding edge cases.
    at::Tensor s_npu = s_cpu.to(like.device(), /*non_blocking=*/false, /*copy=*/true);
    TORCH_CHECK(s_npu.is_contiguous(), "Internal scalar tensor must be contiguous");
    return s_npu;
}

at::Tensor matrix_scalar_multiplication_custom_impl_npu(const at::Tensor& A, double s_double)
{
    TORCH_CHECK(A.defined(), "A must be defined");
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (ND) for matrix_scalar_multiplication_custom");
    TORCH_CHECK(A.numel() >= 0, "A must be a valid tensor");

    at::Tensor S = MakeScalarTensorOnNpuLike(A, s_double);
    TORCH_CHECK(S.scalar_type() == at::kFloat, "Internal scalar tensor must be float32");
    TORCH_CHECK(S.device().type() == c10::DeviceType::PrivateUse1, "Internal scalar tensor must be on NPU");
    TORCH_CHECK(S.numel() == 1, "Internal scalar tensor must have exactly 1 element");

    at::Tensor C = at::empty_like(A);

    EXEC_NPU_CMD(aclnnMatrixScalarMultiplicationCustom, A, S, C);
    return C;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matrix_scalar_multiplication_custom", &matrix_scalar_multiplication_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_scalar_multiplication_custom",
          &matrix_scalar_multiplication_custom_impl_npu,
          "Matrix-scalar multiplication: C = A * s (custom AscendC, NPU)");
}
