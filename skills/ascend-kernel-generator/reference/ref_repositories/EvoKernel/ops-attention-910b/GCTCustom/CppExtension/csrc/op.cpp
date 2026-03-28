
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gct_impl_npu(const at::Tensor& x, double c, double eps) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "gct_custom: x must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "gct_custom: x must be float32");
    TORCH_CHECK(x.dim() == 4, "gct_custom: x must be [N,C,H,W]");
    TORCH_CHECK(x.is_contiguous(), "gct_custom: x must be contiguous (NCHW)");

    // c/eps as 0-dim float tensors on NPU (shape size == 1)
    auto c_t = at::empty({}, x.options());
    c_t.fill_(static_cast<float>(c));
    auto eps_t = at::empty({}, x.options());
    eps_t.fill_(static_cast<float>(eps));

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnGCTCustom, x, c_t, eps_t, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gct_custom", &gct_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gct_custom", &gct_impl_npu, "Gaussian Context Transformer fused op (NPU)");
}
