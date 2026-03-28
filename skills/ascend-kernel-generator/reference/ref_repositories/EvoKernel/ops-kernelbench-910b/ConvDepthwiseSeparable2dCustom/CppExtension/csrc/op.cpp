
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_npu_f32_contig_4d(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D (NCHW / OIHW)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor conv_depthwise_separable2d_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& w_depthwise,
    const at::Tensor& w_pointwise)
{
    check_npu_f32_contig_4d(x, "x");
    check_npu_f32_contig_4d(w_depthwise, "w_depthwise");
    check_npu_f32_contig_4d(w_pointwise, "w_pointwise");

    TORCH_CHECK(x.size(0) == 16, "specialized kernel expects N=16");
    TORCH_CHECK(x.size(1) == 64, "specialized kernel expects Cin=64");
    TORCH_CHECK(x.size(2) == 512 && x.size(3) == 512, "specialized kernel expects H=W=512");

    TORCH_CHECK(w_depthwise.size(0) == 64, "specialized kernel expects w_depthwise[0]=64");
    TORCH_CHECK(w_depthwise.size(1) == 1, "specialized kernel expects depthwise weight shape [64,1,3,3]");
    TORCH_CHECK(w_depthwise.size(2) == 3 && w_depthwise.size(3) == 3, "specialized kernel expects depthwise 3x3");

    TORCH_CHECK(w_pointwise.size(0) == 128, "specialized kernel expects Cout=128");
    TORCH_CHECK(w_pointwise.size(1) == 64, "specialized kernel expects w_pointwise[1]=64");
    TORCH_CHECK(w_pointwise.size(2) == 1 && w_pointwise.size(3) == 1, "specialized kernel expects pointwise 1x1");

    auto y = at::empty({16, 128, 512, 512}, x.options());
    EXEC_NPU_CMD(aclnnConvDepthwiseSeparable2dCustom, x, w_depthwise, w_pointwise, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_depthwise_separable2d_custom",
           &conv_depthwise_separable2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_depthwise_separable2d_custom",
          &conv_depthwise_separable2d_custom_impl_npu,
          "conv_depthwise_separable2d_custom (NPU)");
}
