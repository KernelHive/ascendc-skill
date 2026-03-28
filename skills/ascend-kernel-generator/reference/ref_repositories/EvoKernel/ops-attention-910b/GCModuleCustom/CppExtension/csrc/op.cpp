
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gc_module_impl_npu(const at::Tensor& x, const at::Tensor& y_bc11) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "gc_module_custom: x must be on NPU");
    TORCH_CHECK(y_bc11.device().type() == c10::DeviceType::PrivateUse1, "gc_module_custom: y_bc11 must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "gc_module_custom: only float32 supported");
    TORCH_CHECK(y_bc11.scalar_type() == at::kFloat, "gc_module_custom: only float32 supported");

    TORCH_CHECK(x.dim() == 4, "gc_module_custom: x must be 4D [B,C,H,W]");
    TORCH_CHECK(y_bc11.dim() == 4, "gc_module_custom: y_bc11 must be 4D [B,C,1,1]");

    TORCH_CHECK(x.is_contiguous(), "gc_module_custom: x must be contiguous (NCHW)");
    TORCH_CHECK(y_bc11.is_contiguous(), "gc_module_custom: y_bc11 must be contiguous (NCHW)");

    TORCH_CHECK(y_bc11.size(0) == x.size(0) &&
                y_bc11.size(1) == x.size(1) &&
                y_bc11.size(2) == 1 &&
                y_bc11.size(3) == 1,
                "gc_module_custom: y_bc11 must be [B,C,1,1] matching x's B,C");

    at::Tensor out = at::empty_like(x);
    EXEC_NPU_CMD(aclnnGCModuleCustom, x, y_bc11, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gc_module_custom", &gc_module_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gc_module_custom", &gc_module_impl_npu,
          "GCModule fused broadcast add: out = x + y (y is [B,C,1,1]) on NPU");
}
