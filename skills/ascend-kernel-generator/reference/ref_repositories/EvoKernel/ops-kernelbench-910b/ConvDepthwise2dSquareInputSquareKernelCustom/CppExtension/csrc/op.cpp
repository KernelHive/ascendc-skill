
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_depthwise2d_square_input_square_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C,1,KH,KW] for depthwise");

    TORCH_CHECK(x.size(0) == 16, "specialized kernel expects N=16");
    TORCH_CHECK(x.size(1) == 64, "specialized kernel expects C=64");
    TORCH_CHECK(x.size(2) == 512, "specialized kernel expects H=512");
    TORCH_CHECK(x.size(3) == 512, "specialized kernel expects W=512");

    TORCH_CHECK(weight.size(0) == 64, "specialized kernel expects weight.size(0)=64");
    TORCH_CHECK(weight.size(1) == 1, "depthwise weight must have second dim = 1");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "specialized kernel expects KH=KW=3");

    auto y = at::empty({16, 64, 510, 510}, x.options());
    EXEC_NPU_CMD(aclnnConvDepthwise2dSquareInputSquareKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_depthwise2d_square_input_square_kernel_custom",
           &conv_depthwise2d_square_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_depthwise2d_square_input_square_kernel_custom",
          &conv_depthwise2d_square_input_square_kernel_custom_impl_npu,
          "conv_depthwise2d_square_input_square_kernel_custom (NPU)");
}
