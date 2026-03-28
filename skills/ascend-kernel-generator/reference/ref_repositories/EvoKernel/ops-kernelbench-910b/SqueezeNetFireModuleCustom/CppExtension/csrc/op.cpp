
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

at::Tensor squeeze_net_fire_module_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& w_squeeze, const at::Tensor& b_squeeze,
    const at::Tensor& w_expand1, const at::Tensor& b_expand1,
    const at::Tensor& w_expand3, const at::Tensor& b_expand3)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w_squeeze.device().type() == c10::DeviceType::PrivateUse1, "w_squeeze must be on NPU");
    TORCH_CHECK(b_squeeze.device().type() == c10::DeviceType::PrivateUse1, "b_squeeze must be on NPU");
    TORCH_CHECK(w_expand1.device().type() == c10::DeviceType::PrivateUse1, "w_expand1 must be on NPU");
    TORCH_CHECK(b_expand1.device().type() == c10::DeviceType::PrivateUse1, "b_expand1 must be on NPU");
    TORCH_CHECK(w_expand3.device().type() == c10::DeviceType::PrivateUse1, "w_expand3 must be on NPU");
    TORCH_CHECK(b_expand3.device().type() == c10::DeviceType::PrivateUse1, "b_expand3 must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w_squeeze.scalar_type() == at::kFloat, "w_squeeze must be float32");
    TORCH_CHECK(b_squeeze.scalar_type() == at::kFloat, "b_squeeze must be float32");
    TORCH_CHECK(w_expand1.scalar_type() == at::kFloat, "w_expand1 must be float32");
    TORCH_CHECK(b_expand1.scalar_type() == at::kFloat, "b_expand1 must be float32");
    TORCH_CHECK(w_expand3.scalar_type() == at::kFloat, "w_expand3 must be float32");
    TORCH_CHECK(b_expand3.scalar_type() == at::kFloat, "b_expand3 must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(w_squeeze.is_contiguous(), "w_squeeze must be contiguous");
    TORCH_CHECK(b_squeeze.is_contiguous(), "b_squeeze must be contiguous");
    TORCH_CHECK(w_expand1.is_contiguous(), "w_expand1 must be contiguous");
    TORCH_CHECK(b_expand1.is_contiguous(), "b_expand1 must be contiguous");
    TORCH_CHECK(w_expand3.is_contiguous(), "w_expand3 must be contiguous");
    TORCH_CHECK(b_expand3.is_contiguous(), "b_expand3 must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(w_squeeze.dim() == 4 && w_expand1.dim() == 4 && w_expand3.dim() == 4, "weights must be 4D OIHW");
    TORCH_CHECK(b_squeeze.dim() == 1 && b_expand1.dim() == 1 && b_expand3.dim() == 1, "biases must be 1D");

    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 3 && x.size(2) == 256 && x.size(3) == 256,
                "specialized kernel expects x: [128,3,256,256]");

    TORCH_CHECK(w_squeeze.size(0) == 6 && w_squeeze.size(1) == 3 && w_squeeze.size(2) == 1 && w_squeeze.size(3) == 1,
                "specialized kernel expects w_squeeze: [6,3,1,1]");
    TORCH_CHECK(b_squeeze.size(0) == 6, "specialized kernel expects b_squeeze: [6]");

    TORCH_CHECK(w_expand1.size(0) == 64 && w_expand1.size(1) == 6 && w_expand1.size(2) == 1 && w_expand1.size(3) == 1,
                "specialized kernel expects w_expand1: [64,6,1,1]");
    TORCH_CHECK(b_expand1.size(0) == 64, "specialized kernel expects b_expand1: [64]");

    TORCH_CHECK(w_expand3.size(0) == 64 && w_expand3.size(1) == 6 && w_expand3.size(2) == 3 && w_expand3.size(3) == 3,
                "specialized kernel expects w_expand3: [64,6,3,3]");
    TORCH_CHECK(b_expand3.size(0) == 64, "specialized kernel expects b_expand3: [64]");

    at::Tensor y = at::empty({128, 128, 256, 256}, x.options());

    EXEC_NPU_CMD(aclnnSqueezeNetFireModuleCustom,
                 x,
                 w_squeeze, b_squeeze,
                 w_expand1, b_expand1,
                 w_expand3, b_expand3,
                 y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("squeeze_net_fire_module_custom", &squeeze_net_fire_module_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("squeeze_net_fire_module_custom",
          &squeeze_net_fire_module_custom_impl_npu,
          "squeeze_net_fire_module_custom (NPU)");
}
