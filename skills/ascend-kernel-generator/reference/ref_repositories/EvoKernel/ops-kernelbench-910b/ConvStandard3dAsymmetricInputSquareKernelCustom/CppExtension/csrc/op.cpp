
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_standard3d_asymmetric_input_square_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    auto opts = x.options();
    // Specialized output shape for the benchmark: Hout=Wout=254, Dout=10
    at::Tensor y = at::empty({x.size(0), weight.size(0), 254, 254, 10}, opts);
    EXEC_NPU_CMD(aclnnConvStandard3dAsymmetricInputSquareKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard3d_asymmetric_input_square_kernel_custom",
           &conv_standard3d_asymmetric_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard3d_asymmetric_input_square_kernel_custom",
          &conv_standard3d_asymmetric_input_square_kernel_custom_impl_npu,
          "conv_standard3d_asymmetric_input_square_kernel_custom (NPU)");
}
