
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C,1,Kh,Kw)");
    TORCH_CHECK(weight.size(1) == 1, "depthwise weight second dim must be 1");
    TORCH_CHECK(x.size(1) == weight.size(0), "C mismatch: x.size(1) must equal weight.size(0)");

    // Specialized parameters: Kh=3, Kw=7, stride=1, pad=0, dilation=1
    TORCH_CHECK(weight.size(2) == 3, "specialized op requires Kh=3");
    TORCH_CHECK(weight.size(3) == 7, "specialized op requires Kw=7");

    TORCH_CHECK(x.size(0) == 32, "specialized op requires N=32");
    TORCH_CHECK(x.size(1) == 128, "specialized op requires C=128");
    TORCH_CHECK(x.size(2) == 128, "specialized op requires H=128");
    TORCH_CHECK(x.size(3) == 256, "specialized op requires W=256");

    // Output shape for pad=0, stride=1, dilation=1:
    // Ho = H - (Kh - 1) - 1 + 1 = H - Kh + 1 = 126
    // Wo = W - Kw + 1 = 250
    auto opts = x.options();
    at::Tensor y = at::empty({32, 128, 126, 250}, opts);

    EXEC_NPU_CMD(aclnnConvDepthwise2dAsymmetricInputAsymmetricKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom",
           &conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom",
          &conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_impl_npu,
          "conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom (NPU)");
}
