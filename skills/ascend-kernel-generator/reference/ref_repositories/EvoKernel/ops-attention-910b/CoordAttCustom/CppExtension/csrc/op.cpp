
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor coord_att_impl_npu(const at::Tensor& x,
                             const at::Tensor& a_w,
                             const at::Tensor& a_h) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "coord_att_custom: only float32 supported");
    TORCH_CHECK(a_w.scalar_type() == at::kFloat, "coord_att_custom: only float32 supported");
    TORCH_CHECK(a_h.scalar_type() == at::kFloat, "coord_att_custom: only float32 supported");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "coord_att_custom: x must be on NPU");
    TORCH_CHECK(a_w.device().type() == c10::DeviceType::PrivateUse1, "coord_att_custom: a_w must be on NPU");
    TORCH_CHECK(a_h.device().type() == c10::DeviceType::PrivateUse1, "coord_att_custom: a_h must be on NPU");

    TORCH_CHECK(x.is_contiguous(), "coord_att_custom: x must be contiguous");
    TORCH_CHECK(a_w.is_contiguous(), "coord_att_custom: a_w must be contiguous");
    TORCH_CHECK(a_h.is_contiguous(), "coord_att_custom: a_h must be contiguous");

    TORCH_CHECK(x.sizes() == a_w.sizes(), "coord_att_custom: shape mismatch (x vs a_w)");
    TORCH_CHECK(x.sizes() == a_h.sizes(), "coord_att_custom: shape mismatch (x vs a_h)");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnCoordAttCustom, x, a_w, a_h, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("coord_att_custom", &coord_att_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coord_att_custom", &coord_att_impl_npu,
          "CoordAtt fused tail: y = x * a_w * a_h (NPU, float32)");
}
