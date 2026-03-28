
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convtranspose3d_out(int64_t in, int64_t stride, int64_t pad, int64_t dil, int64_t k, int64_t out_pad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

static inline void check_5d_ncdhw_float_contig_npu(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 5, name, " must be 5D NCDHW");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (NCDHW)");
}

at::Tensor conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_5d_ncdhw_float_contig_npu(x, "x");
    check_5d_ncdhw_float_contig_npu(weight, "weight");
    TORCH_CHECK(weight.dim() == 5, "weight must be [Cin, Cout/groups, Kd, Kh, Kw] 5D");

    // Fixed params from the target model instance
    const int64_t stride_d = 2, stride_h = 2, stride_w = 2;
    const int64_t pad_d = 1, pad_h = 2, pad_w = 3;
    const int64_t outpad_d = 1, outpad_h = 1, outpad_w = 1;
    const int64_t dil_d = 1, dil_h = 1, dil_w = 1;
    const int64_t groups = 4;

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Din = x.size(2);
    const int64_t Hin = x.size(3);
    const int64_t Win = x.size(4);

    TORCH_CHECK(N == 8 && Cin == 32 && Din == 12 && Hin == 24 && Win == 48,
                "specialized for x=[8,32,12,24,48]");

    const int64_t wCin = weight.size(0);
    const int64_t wCoutPerG = weight.size(1);
    const int64_t Kd = weight.size(2);
    const int64_t Kh = weight.size(3);
    const int64_t Kw = weight.size(4);

    TORCH_CHECK(wCin == 32 && wCoutPerG == 8 && Kd == 3 && Kh == 5 && Kw == 7,
                "specialized for weight=[32,8,3,5,7]");

    TORCH_CHECK(groups > 0, "groups must be > 0");
    TORCH_CHECK(Cin % groups == 0, "Cin must be divisible by groups");

    const int64_t Cout = wCoutPerG * groups; // 32

    const int64_t Dout = convtranspose3d_out(Din, stride_d, pad_d, dil_d, Kd, outpad_d); // 24
    const int64_t Hout = convtranspose3d_out(Hin, stride_h, pad_h, dil_h, Kh, outpad_h); // 48
    const int64_t Wout = convtranspose3d_out(Win, stride_w, pad_w, dil_w, Kw, outpad_w); // 96

    TORCH_CHECK(Dout == 24 && Hout == 48 && Wout == 96, "unexpected output shape for specialization");
    TORCH_CHECK(Cout == 32, "unexpected Cout for specialization");

    at::Tensor y = at::empty({N, Cout, Dout, Hout, Wout}, x.options());

    EXEC_NPU_CMD(aclnnConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom,
                 x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom",
           &conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom",
          &conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom_impl_npu,
          "conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom (NPU)");
}
