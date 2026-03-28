
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convtranspose1d_out(int64_t Lin, int64_t stride, int64_t pad, int64_t dil, int64_t k)
{
    // output_padding = 0
    return (Lin - 1) * stride - 2 * pad + dil * (k - 1) + 1;
}

at::Tensor conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous [N,C,L]");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous [Cin,Cout,K]");
    TORCH_CHECK(x.dim() == 3, "x must be [N, Cin, Lin]");
    TORCH_CHECK(weight.dim() == 3, "weight must be [Cin, Cout, K] (groups=1)");

    // Specialized fixed params
    const int64_t stride = 2;
    const int64_t pad = 1;
    const int64_t dil = 2;

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Lin = x.size(2);

    const int64_t wCin = weight.size(0);
    const int64_t wCout = weight.size(1);
    const int64_t K = weight.size(2);

    TORCH_CHECK(N == 16 && Cin == 32 && Lin == 131072,
                "specialized for x=[16,32,131072]");
    TORCH_CHECK(wCin == 32 && wCout == 64 && K == 3,
                "specialized for weight=[32,64,3] (PyTorch ConvTranspose1d layout)");

    const int64_t Lout = convtranspose1d_out(Lin, stride, pad, dil, K); // 262145
    TORCH_CHECK(Lout == 262145, "specialized kernel expects Lout=262145");

    auto y = at::empty({N, wCout, Lout}, x.options());

    EXEC_NPU_CMD(aclnnConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom,
                 x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom",
           &conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom",
          &conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_impl_npu,
          "conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom (NPU)");
}
