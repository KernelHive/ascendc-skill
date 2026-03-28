
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_min_softmax_custom_impl_npu(const at::Tensor& x,
                                             const at::Tensor& weight,
                                             const at::Tensor& bias)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cout,Cin,Kd,Kh,Kw]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [Cout]");

    // Fixed specialization contract for this custom kernel.
    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 3 && x.size(2) == 24 && x.size(3) == 32 && x.size(4) == 32,
                "x shape must be [128,3,24,32,32] for conv3d_min_softmax_custom");
    TORCH_CHECK(weight.size(0) == 24 && weight.size(1) == 3 &&
                weight.size(2) == 3 && weight.size(3) == 3 && weight.size(4) == 3,
                "weight shape must be [24,3,3,3,3] for conv3d_min_softmax_custom");
    TORCH_CHECK(bias.size(0) == 24, "bias shape must be [24] for conv3d_min_softmax_custom");

    auto y = at::empty({128, 24, 30, 30}, x.options());
    EXEC_NPU_CMD(aclnnConv3dMinSoftmaxCustom, x, weight, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_min_softmax_custom", &conv3d_min_softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_min_softmax_custom", &conv3d_min_softmax_custom_impl_npu,
          "conv3d_min_softmax_custom (NPU)");
}
