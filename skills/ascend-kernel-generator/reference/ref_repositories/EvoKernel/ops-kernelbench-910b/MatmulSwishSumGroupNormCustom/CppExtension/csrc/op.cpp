
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_swish_sum_group_norm_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& w,            // [N,K] Linear weight [out,in]
    const at::Tensor& linear_bias,  // [N]
    const at::Tensor& add_bias,     // [N]
    const at::Tensor& gamma,        // [N]
    const at::Tensor& beta,         // [N]
    const at::Tensor& num_groups,   // scalar int32 tensor (numel==1)
    const at::Tensor& eps)          // scalar float tensor (numel==1)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "w must be on NPU");
    TORCH_CHECK(linear_bias.device().type() == c10::DeviceType::PrivateUse1, "linear_bias must be on NPU");
    TORCH_CHECK(add_bias.device().type() == c10::DeviceType::PrivateUse1, "add_bias must be on NPU");
    TORCH_CHECK(gamma.device().type() == c10::DeviceType::PrivateUse1, "gamma must be on NPU");
    TORCH_CHECK(beta.device().type() == c10::DeviceType::PrivateUse1, "beta must be on NPU");
    TORCH_CHECK(num_groups.device().type() == c10::DeviceType::PrivateUse1, "num_groups must be on NPU");
    TORCH_CHECK(eps.device().type() == c10::DeviceType::PrivateUse1, "eps must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(linear_bias.scalar_type() == at::kFloat, "linear_bias must be float32");
    TORCH_CHECK(add_bias.scalar_type() == at::kFloat, "add_bias must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
    TORCH_CHECK(num_groups.scalar_type() == at::kInt, "num_groups must be int32");
    TORCH_CHECK(eps.scalar_type() == at::kFloat, "eps must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(linear_bias.dim() == 1, "linear_bias must be 1D [N]");
    TORCH_CHECK(add_bias.dim() == 1, "add_bias must be 1D [N]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [N]");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D [N]");
    TORCH_CHECK(num_groups.numel() == 1, "num_groups must be a scalar tensor (numel==1)");
    TORCH_CHECK(eps.numel() == 1, "eps must be a scalar tensor (numel==1)");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(linear_bias.is_contiguous(), "linear_bias must be contiguous");
    TORCH_CHECK(add_bias.is_contiguous(), "add_bias must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(num_groups.is_contiguous(), "num_groups must be contiguous");
    TORCH_CHECK(eps.is_contiguous(), "eps must be contiguous");

    // Fixed specialized contract (matches host tiling & kernel assumptions).
    TORCH_CHECK(x.size(0) == 32768 && x.size(1) == 1024,
                "x shape must be [32768,1024] for matmul_swish_sum_group_norm_custom");
    TORCH_CHECK(w.size(0) == 4096 && w.size(1) == 1024,
                "w shape must be [4096,1024] (out,in) for matmul_swish_sum_group_norm_custom");
    TORCH_CHECK(linear_bias.size(0) == 4096, "linear_bias shape must be [4096]");
    TORCH_CHECK(add_bias.size(0) == 4096, "add_bias shape must be [4096]");
    TORCH_CHECK(gamma.size(0) == 4096, "gamma shape must be [4096]");
    TORCH_CHECK(beta.size(0) == 4096, "beta shape must be [4096]");

    // Enforce specialization on num_groups value (host-side check only).
    int32_t ng = num_groups.to(at::kCPU).item<int32_t>();
    TORCH_CHECK(ng == 64, "num_groups must be 64 for matmul_swish_sum_group_norm_custom");

    auto y = at::empty({32768, 4096}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMatmulSwishSumGroupNormCustom,
                 x, w, linear_bias, add_bias, gamma, beta, num_groups, eps, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_swish_sum_group_norm_custom",
           &matmul_swish_sum_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_swish_sum_group_norm_custom",
          &matmul_swish_sum_group_norm_custom_impl_npu,
          "matmul_swish_sum_group_norm_custom (NPU)");
}
