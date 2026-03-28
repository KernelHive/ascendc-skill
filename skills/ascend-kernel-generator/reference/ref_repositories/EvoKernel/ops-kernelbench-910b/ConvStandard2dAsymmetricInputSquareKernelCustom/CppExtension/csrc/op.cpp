
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_npu_f32_contig_4d(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor conv_standard2d_asymmetric_input_square_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_npu_f32_contig_4d(x, "x");
    check_npu_f32_contig_4d(weight, "weight");

    TORCH_CHECK(x.size(0) == 8, "specialized kernel expects N=8");
    TORCH_CHECK(x.size(1) == 64, "specialized kernel expects Cin=64");
    TORCH_CHECK(x.size(2) == 512, "specialized kernel expects H=512");
    TORCH_CHECK(x.size(3) == 1024, "specialized kernel expects W=1024");

    TORCH_CHECK(weight.size(0) == 128, "specialized kernel expects Cout=128");
    TORCH_CHECK(weight.size(1) == 64, "specialized kernel expects Cin=64");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "specialized kernel expects KH=KW=3");

    auto y = at::empty({8, 128, 510, 1022}, x.options());

    EXEC_NPU_CMD(aclnnConvStandard2dAsymmetricInputSquareKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard2d_asymmetric_input_square_kernel_custom",
           &conv_standard2d_asymmetric_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard2d_asymmetric_input_square_kernel_custom",
          &conv_standard2d_asymmetric_input_square_kernel_custom_impl_npu,
          "conv_standard2d_asymmetric_input_square_kernel_custom (NPU)");
}
