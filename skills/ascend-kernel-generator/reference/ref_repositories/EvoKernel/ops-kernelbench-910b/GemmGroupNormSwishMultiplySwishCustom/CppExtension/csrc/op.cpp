
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_group_norm_swish_multiply_swish_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    const at::Tensor& mul_w,
    const at::Tensor& num_groups,
    const at::Tensor& eps)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "w must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "b must be on NPU");
    TORCH_CHECK(gamma.device().type() == c10::DeviceType::PrivateUse1, "gamma must be on NPU");
    TORCH_CHECK(beta.device().type() == c10::DeviceType::PrivateUse1, "beta must be on NPU");
    TORCH_CHECK(mul_w.device().type() == c10::DeviceType::PrivateUse1, "mul_w must be on NPU");
    TORCH_CHECK(num_groups.device().type() == c10::DeviceType::PrivateUse1, "num_groups must be on NPU");
    TORCH_CHECK(eps.device().type() == c10::DeviceType::PrivateUse1, "eps must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
    TORCH_CHECK(mul_w.scalar_type() == at::kFloat, "mul_w must be float32");
    TORCH_CHECK(num_groups.scalar_type() == at::kInt, "num_groups must be int32");
    TORCH_CHECK(eps.scalar_type() == at::kFloat, "eps must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [N]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [N]");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D [N]");
    TORCH_CHECK(mul_w.dim() == 1, "mul_w must be 1D [N]");

    TORCH_CHECK(num_groups.numel() == 1, "num_groups must be a scalar tensor (numel==1)");
    TORCH_CHECK(eps.numel() == 1, "eps must be a scalar tensor (numel==1)");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(mul_w.is_contiguous(), "mul_w must be contiguous");
    TORCH_CHECK(num_groups.is_contiguous(), "num_groups must be contiguous");
    TORCH_CHECK(eps.is_contiguous(), "eps must be contiguous");

    // Fixed specialized contract (matches host tiling & kernel assumptions).
    TORCH_CHECK(x.size(0) == 1024 && x.size(1) == 8192,
                "x shape must be [1024,8192] for gemm_group_norm_swish_multiply_swish_custom");
    TORCH_CHECK(w.size(0) == 8192 && w.size(1) == 8192,
                "w shape must be [8192,8192] (out,in) for gemm_group_norm_swish_multiply_swish_custom");
    TORCH_CHECK(b.size(0) == 8192, "b shape must be [8192]");
    TORCH_CHECK(gamma.size(0) == 8192, "gamma shape must be [8192]");
    TORCH_CHECK(beta.size(0) == 8192, "beta shape must be [8192]");
    TORCH_CHECK(mul_w.size(0) == 8192, "mul_w shape must be [8192]");

    // Enforce specialization on num_groups value. Copying scalar to CPU is acceptable (host-side check only).
    int32_t ng = num_groups.to(at::kCPU).item<int32_t>();
    TORCH_CHECK(ng == 256, "num_groups must be 256 for gemm_group_norm_swish_multiply_swish_custom");

    auto y = at::empty({1024, 8192}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnGemmGroupNormSwishMultiplySwishCustom,
                 x, w, b, gamma, beta, mul_w, num_groups, eps, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_group_norm_swish_multiply_swish_custom",
           &gemm_group_norm_swish_multiply_swish_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_group_norm_swish_multiply_swish_custom",
          &gemm_group_norm_swish_multiply_swish_custom_impl_npu,
          "gemm_group_norm_swish_multiply_swish_custom (NPU)");
}
