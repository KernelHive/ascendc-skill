
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t conv_out_dim(int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride) {
    int64_t numer = in + 2 * pad - dil * (k - 1) - 1;
    return numer / stride + 1;
}

at::Tensor conv_standard3d_square_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,W,H,D] for this specialized op");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cout,Cin,Kw,Kh,Kd]");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t W   = x.size(2);
    const int64_t H   = x.size(3);
    const int64_t D   = x.size(4);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kw   = weight.size(2);
    const int64_t Kh   = weight.size(3);
    const int64_t Kd   = weight.size(4);

    TORCH_CHECK(N == 16 && Cin == 3 && W == 64 && H == 64 && D == 64,
                "specialized op requires x shape [16,3,64,64,64] in layout [N,C,W,H,D]");
    TORCH_CHECK(Cout == 64 && wCin == 3 && Kw == 3 && Kh == 5 && Kd == 7,
                "specialized op requires weight shape [64,3,3,5,7]");
    TORCH_CHECK(Cin == wCin, "Cin mismatch between x and weight");

    const int64_t stride = 1, pad = 0, dil = 1;
    const int64_t Wout = conv_out_dim(W, pad, dil, Kw, stride);
    const int64_t Hout = conv_out_dim(H, pad, dil, Kh, stride);
    const int64_t Dout = conv_out_dim(D, pad, dil, Kd, stride);
    TORCH_CHECK(Wout == 62 && Hout == 60 && Dout == 58, "unexpected output shape for specialization");

    at::Tensor y = at::empty({N, Cout, Wout, Hout, Dout}, x.options());
    EXEC_NPU_CMD(aclnnConvStandard3dSquareInputAsymmetricKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard3d_square_input_asymmetric_kernel_custom",
           &conv_standard3d_square_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard3d_square_input_asymmetric_kernel_custom",
          &conv_standard3d_square_input_asymmetric_kernel_custom_impl_npu,
          "conv_standard3d_square_input_asymmetric_kernel_custom (NPU, specialized)");
}
