
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t conv_out_size(int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride)
{
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

static inline void check_npu_f32_contig_4d(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D (NCHW / OIHW)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_npu_f32_contig_4d(x, "x");
    check_npu_f32_contig_4d(weight, "weight");

    TORCH_CHECK(x.size(0) == 8, "specialized kernel expects N=8");
    TORCH_CHECK(x.size(1) == 32, "specialized kernel expects Cin=32");
    TORCH_CHECK(x.size(2) == 512 && x.size(3) == 512, "specialized kernel expects H=W=512");

    TORCH_CHECK(weight.size(0) == 64, "specialized kernel expects Cout=64");
    TORCH_CHECK(weight.size(1) == 32, "specialized kernel expects weight.Cin=32");
    TORCH_CHECK(weight.size(2) == 5 && weight.size(3) == 9, "specialized kernel expects Kh=5, Kw=9");

    constexpr int64_t strideH = 1, strideW = 1;
    constexpr int64_t padH = 2, padW = 4;
    constexpr int64_t dilH = 2, dilW = 3;

    const int64_t Ho = conv_out_size(x.size(2), padH, dilH, weight.size(2), strideH);
    const int64_t Wo = conv_out_size(x.size(3), padW, dilW, weight.size(3), strideW);
    TORCH_CHECK(Ho == 508 && Wo == 496, "specialized kernel expects Ho=508, Wo=496");

    auto y = at::empty({8, 64, Ho, Wo}, x.options());

    EXEC_NPU_CMD(aclnnConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom",
           &conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom",
          &conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_impl_npu,
          "conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom (NPU)");
}
