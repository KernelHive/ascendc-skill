
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_standard3d_square_input_square_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cout,Cin,K,K,K]");

    TORCH_CHECK(x.size(0) == 16 && x.size(1) == 3 &&
                x.size(2) == 64 && x.size(3) == 64 && x.size(4) == 64,
                "x shape must be [16,3,64,64,64] for this custom op");
    TORCH_CHECK(weight.size(0) == 64 && weight.size(1) == 3 &&
                weight.size(2) == 3 && weight.size(3) == 3 && weight.size(4) == 3,
                "weight shape must be [64,3,3,3,3] for this custom op");

    auto y = at::empty({16, 64, 62, 62, 62}, x.options());
    EXEC_NPU_CMD(aclnnConvStandard3dSquareInputSquareKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard3d_square_input_square_kernel_custom",
           &conv_standard3d_square_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard3d_square_input_square_kernel_custom",
          &conv_standard3d_square_input_square_kernel_custom_impl_npu,
          "conv_standard3d_square_input_square_kernel_custom (NPU)");
}
