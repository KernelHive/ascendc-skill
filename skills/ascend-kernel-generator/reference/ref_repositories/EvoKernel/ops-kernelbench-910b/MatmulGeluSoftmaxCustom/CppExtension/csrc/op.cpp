
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_gelu_softmax_custom_impl_npu(const at::Tensor& x,
                                              const at::Tensor& weight,
                                              const at::Tensor& bias) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, K)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (N, K) like nn.Linear.weight");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (N)");

    TORCH_CHECK(x.numel() > 0, "x must be non-empty");
    TORCH_CHECK(weight.numel() > 0, "weight must be non-empty");
    TORCH_CHECK(bias.numel() > 0, "bias must be non-empty");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(weight.size(1) == x.size(1), "K mismatch: weight.size(1) must equal x.size(1)");
    TORCH_CHECK(bias.size(0) == weight.size(0), "N mismatch: bias.size(0) must equal weight.size(0)");

    const auto B = x.size(0);
    const auto N = weight.size(0);

    at::Tensor y = at::empty({B, N}, x.options());
    EXEC_NPU_CMD(aclnnMatmulGeluSoftmaxCustom, x, weight, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_gelu_softmax_custom", &matmul_gelu_softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_gelu_softmax_custom",
          &matmul_gelu_softmax_custom_impl_npu,
          "Matmul+Bias+GELU+Softmax(dim=1) fused (NPU)");
}
