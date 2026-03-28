
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_mish_tanh_custom_impl_npu(const at::Tensor& x)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D NCDHW");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (ND/NCDHW)");

    // output same shape
    at::Tensor y = at::empty_like(x);

    // Custom op: single input, single output (must match opdef/json/kernel)
    EXEC_NPU_CMD(aclnnConv3dMishTanhCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_mish_tanh_custom", &conv3d_mish_tanh_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_mish_tanh_custom",
          &conv3d_mish_tanh_custom_impl_npu,
          "conv3d_mish_tanh_custom (NPU)");
}
