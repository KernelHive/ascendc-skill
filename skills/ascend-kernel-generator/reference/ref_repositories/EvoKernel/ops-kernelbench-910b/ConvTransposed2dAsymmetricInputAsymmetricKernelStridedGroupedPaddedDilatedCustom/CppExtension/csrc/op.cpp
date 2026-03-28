
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convtranspose2d_out(int64_t in, int64_t stride, int64_t pad, int64_t dil, int64_t k)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + 1;
}

at::Tensor conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cin, Cout/groups, Kh, Kw] 4D");

    // Fixed params from the target model instance
    const int64_t stride_h = 2, stride_w = 3;
    const int64_t pad_h = 1, pad_w = 2;
    const int64_t dilation_h = 2, dilation_w = 1;
    const int64_t groups = 4;

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    TORCH_CHECK(N == 16 && Cin == 32 && Hin == 128 && Win == 256,
                "specialized for x=[16,32,128,256]");

    const int64_t wCin = weight.size(0);
    const int64_t wCoutPerG = weight.size(1);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(wCin == 32 && wCoutPerG == 16 && Kh == 3 && Kw == 5,
                "specialized for weight=[32,16,3,5]");

    TORCH_CHECK(groups > 0, "groups must be > 0");
    TORCH_CHECK(Cin % groups == 0, "Cin must be divisible by groups");

    const int64_t Cout = wCoutPerG * groups; // 64
    const int64_t Hout = convtranspose2d_out(Hin, stride_h, pad_h, dilation_h, Kh); // 255
    const int64_t Wout = convtranspose2d_out(Win, stride_w, pad_w, dilation_w, Kw); // 766

    at::Tensor y = at::empty({N, Cout, Hout, Wout}, x.options());

    EXEC_NPU_CMD(aclnnConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom,
                 x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom",
           &conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom",
          &conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom_impl_npu,
          "conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom (NPU)");
}
