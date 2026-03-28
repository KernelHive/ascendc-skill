
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convtranspose2d_out(int64_t in, int64_t stride, int64_t pad, int64_t dil, int64_t k)
{
    // output_padding = 0
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + 1;
}

static inline void check_4d_nchw_float_contig_npu(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D NCHW");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (NCHW)");
}

at::Tensor conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_4d_nchw_float_contig_npu(x, "x");
    check_4d_nchw_float_contig_npu(weight, "weight");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cin, Cout, Kh, Kw] 4D (groups=1)");

    // Specialized fixed params
    const int64_t stride_h = 5, stride_w = 5;
    const int64_t pad_h = 1, pad_w = 1;
    const int64_t dilation_h = 2, dilation_w = 2;

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    const int64_t wCin = weight.size(0);
    const int64_t wCout = weight.size(1);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(N == 16 && Cin == 32 && Hin == 64 && Win == 128,
                "specialized for x=[16,32,64,128]");
    TORCH_CHECK(wCin == 32 && wCout == 64 && Kh == 3 && Kw == 3,
                "specialized for weight=[32,64,3,3] (PyTorch ConvTranspose2d layout)");

    const int64_t Hout = convtranspose2d_out(Hin, stride_h, pad_h, dilation_h, Kh); // 318
    const int64_t Wout = convtranspose2d_out(Win, stride_w, pad_w, dilation_w, Kw); // 638
    TORCH_CHECK(Hout == 318 && Wout == 638, "specialized kernel expects Hout=318,Wout=638");

    auto y = at::empty({N, wCout, Hout, Wout}, x.options());

    EXEC_NPU_CMD(aclnnConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom,
                 x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom",
           &conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom",
          &conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom_impl_npu,
          "conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom (NPU)");
}
