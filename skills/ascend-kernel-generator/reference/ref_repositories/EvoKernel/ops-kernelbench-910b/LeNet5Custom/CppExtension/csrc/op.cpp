
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor le_net5_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& conv1_w, const at::Tensor& conv1_b,
    const at::Tensor& conv2_w, const at::Tensor& conv2_b,
    const at::Tensor& fc1_w,   const at::Tensor& fc1_b,
    const at::Tensor& fc2_w,   const at::Tensor& fc2_b,
    const at::Tensor& fc3_w,   const at::Tensor& fc3_b)
{
    auto opts = x.options();
    at::Tensor y = at::empty({x.size(0), fc3_b.size(0)}, opts);
    EXEC_NPU_CMD(aclnnLeNet5Custom,
                 x,
                 conv1_w, conv1_b,
                 conv2_w, conv2_b,
                 fc1_w, fc1_b,
                 fc2_w, fc2_b,
                 fc3_w, fc3_b,
                 y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("le_net5_custom", &le_net5_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("le_net5_custom", &le_net5_custom_impl_npu, "le_net5_custom fused LeNet-5 forward (NPU)");
}
