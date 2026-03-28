
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor da_module_impl_npu(const at::Tensor& p_out, const at::Tensor& c_out, int64_t h, int64_t w) {
    // p_out: [bs, hw, c]; c_out: [bs, c, hw]
    TORCH_CHECK(p_out.scalar_type() == at::kFloat, "da_module_custom: only float32 supported");
    TORCH_CHECK(c_out.scalar_type() == at::kFloat, "da_module_custom: only float32 supported");
    TORCH_CHECK(p_out.dim() == 3 && c_out.dim() == 3, "da_module_custom: inputs must be 3D");
    TORCH_CHECK(p_out.is_contiguous(), "da_module_custom: p_out must be contiguous");
    TORCH_CHECK(c_out.is_contiguous(), "da_module_custom: c_out must be contiguous");

    auto bs = p_out.size(0);
    auto hw = p_out.size(1);
    auto c  = p_out.size(2);
    TORCH_CHECK(c_out.size(0) == bs && c_out.size(1) == c && c_out.size(2) == hw,
                "da_module_custom: shape mismatch");

    TORCH_CHECK(h > 0 && w > 0, "da_module_custom: h and w must be positive");
    TORCH_CHECK(hw == h * w, "da_module_custom: p_out.size(1) must equal h*w");

    // Output tensor is NCHW contiguous; kernel writes it as flattened [bs,c,hw] which matches NCHW storage.
    at::Tensor y = at::empty({bs, c, h, w}, p_out.options());
    TORCH_CHECK(y.is_contiguous(), "da_module_custom: output must be contiguous");

    EXEC_NPU_CMD(aclnnDAModuleCustom, p_out, c_out, h, w, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("da_module_custom", &da_module_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("da_module_custom", &da_module_impl_npu, "DANet module fusion (p_out[bs,hw,c] + c_out[bs,c,hw] -> y[bs,c,h,w])");
}
