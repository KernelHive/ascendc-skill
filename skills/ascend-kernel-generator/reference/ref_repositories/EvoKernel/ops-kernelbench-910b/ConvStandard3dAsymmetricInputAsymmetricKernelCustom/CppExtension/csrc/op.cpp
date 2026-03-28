
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t conv_out(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil)
{
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

at::Tensor conv_standard3d_asymmetric_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCDHW)");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous (Cout,Cin,KD,KH,KW)");
    TORCH_CHECK(x.dim() == 5, "x must be 5D NCDHW");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (Cout,Cin,KD,KH,KW)");

    TORCH_CHECK(x.size(0) == 8, "specialized kernel expects N=8");
    TORCH_CHECK(x.size(1) == 3, "specialized kernel expects Cin=3");
    TORCH_CHECK(x.size(2) == 16, "specialized kernel expects D=16");
    TORCH_CHECK(x.size(3) == 128, "specialized kernel expects H=128");
    TORCH_CHECK(x.size(4) == 128, "specialized kernel expects W=128");

    TORCH_CHECK(weight.size(0) == 64, "specialized kernel expects Cout=64");
    TORCH_CHECK(weight.size(1) == 3, "specialized kernel expects weight Cin=3");
    TORCH_CHECK(weight.size(2) == 3, "specialized kernel expects KD=3");
    TORCH_CHECK(weight.size(3) == 5, "specialized kernel expects KH=5");
    TORCH_CHECK(weight.size(4) == 7, "specialized kernel expects KW=7");

    const int64_t strideD = 1, strideH = 1, strideW = 1;
    const int64_t padD = 0, padH = 0, padW = 0;
    const int64_t dilD = 1, dilH = 1, dilW = 1;

    const int64_t outD = conv_out(x.size(2), weight.size(2), padD, strideD, dilD);
    const int64_t outH = conv_out(x.size(3), weight.size(3), padH, strideH, dilH);
    const int64_t outW = conv_out(x.size(4), weight.size(4), padW, strideW, dilW);

    TORCH_CHECK(outD == 14 && outH == 124 && outW == 122,
                "specialized kernel expects output shape (14,124,122)");

    auto y = at::empty({x.size(0), weight.size(0), outD, outH, outW}, x.options());
    EXEC_NPU_CMD(aclnnConvStandard3dAsymmetricInputAsymmetricKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard3d_asymmetric_input_asymmetric_kernel_custom",
           &conv_standard3d_asymmetric_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard3d_asymmetric_input_asymmetric_kernel_custom",
          &conv_standard3d_asymmetric_input_asymmetric_kernel_custom_impl_npu,
          "conv_standard3d_asymmetric_input_asymmetric_kernel_custom (NPU)");
}
