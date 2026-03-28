
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_dropout_softmax_custom_impl_npu(const at::Tensor& x,
                                                 const at::Tensor& weight,
                                                 const at::Tensor& bias,
                                                 const at::Tensor& dropout_p,
                                                 const at::Tensor& training) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, K)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (N, K)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (N)");
    TORCH_CHECK(dropout_p.numel() == 1, "dropout_p must be a scalar tensor");
    TORCH_CHECK(training.numel() == 1, "training must be a scalar tensor (int32 0/1)");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");
    TORCH_CHECK(dropout_p.device().type() == c10::DeviceType::PrivateUse1, "dropout_p must be on NPU");
    TORCH_CHECK(training.device().type() == c10::DeviceType::PrivateUse1, "training must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(dropout_p.scalar_type() == at::kFloat, "dropout_p must be float32");
    TORCH_CHECK(training.scalar_type() == at::kInt, "training must be int32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(dropout_p.is_contiguous(), "dropout_p must be contiguous");
    TORCH_CHECK(training.is_contiguous(), "training must be contiguous");

    TORCH_CHECK(weight.size(1) == x.size(1), "K mismatch: weight.size(1) must equal x.size(1)");
    TORCH_CHECK(bias.size(0) == weight.size(0), "N mismatch: bias.size(0) must equal weight.size(0)");

    const auto B = x.size(0);
    const auto N = weight.size(0);

    at::Tensor y = at::empty({B, N}, x.options());
    EXEC_NPU_CMD(aclnnMatmulDropoutSoftmaxCustom, x, weight, bias, dropout_p, training, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_dropout_softmax_custom", &matmul_dropout_softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_dropout_softmax_custom",
          &matmul_dropout_softmax_custom_impl_npu,
          "Matmul+Bias+Dropout(deterministic when training)+Softmax (NPU)");
}
