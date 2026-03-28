
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor instance_norm_custom_impl_npu(const at::Tensor& x) {
    // Fused InstanceNorm2d replacement: per-(N,C) over HxW, affine=False, track_running_stats=False, eps=1e-5.
    // Float32 NCHW only.
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnInstanceNormCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("instance_norm_custom", &instance_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("instance_norm_custom", &instance_norm_custom_impl_npu,
          "instance_norm_custom(x) -> InstanceNorm2d over HxW (affine=False, eps=1e-5, float32 NCHW)");
}
