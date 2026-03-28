
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t deconv_out(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil, int64_t outpad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + outpad + 1;
}

static inline void check_5d_ncdhw_float_contig_npu(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 5, name, " must be 5D NCDHW");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (NCDHW)");
}

at::Tensor conv_transposed3d_asymmetric_input_asymmetric_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_5d_ncdhw_float_contig_npu(x, "x");
    check_5d_ncdhw_float_contig_npu(weight, "weight");

    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cin,Cout,KD,KH,KW]");

    // Specialization contract (must match kernel constants exactly).
    TORCH_CHECK(x.size(0) == 16, "specialized kernel expects N=16");
    TORCH_CHECK(x.size(1) == 32, "specialized kernel expects Cin=32");
    TORCH_CHECK(x.size(2) == 16, "specialized kernel expects Din=16");
    TORCH_CHECK(x.size(3) == 32, "specialized kernel expects Hin=32");
    TORCH_CHECK(x.size(4) == 64, "specialized kernel expects Win=64");

    TORCH_CHECK(weight.size(0) == 32, "specialized kernel expects weight Cin=32");
    TORCH_CHECK(weight.size(1) == 16, "specialized kernel expects weight Cout=16");
    TORCH_CHECK(weight.size(2) == 3,  "specialized kernel expects KD=3");
    TORCH_CHECK(weight.size(3) == 5,  "specialized kernel expects KH=5");
    TORCH_CHECK(weight.size(4) == 7,  "specialized kernel expects KW=7");
    TORCH_CHECK(weight.size(0) == x.size(1), "weight Cin must equal x Cin");

    const int64_t strideD = 1, strideH = 1, strideW = 1;
    const int64_t padD = 0, padH = 0, padW = 0;
    const int64_t outpadD = 0, outpadH = 0, outpadW = 0;
    const int64_t dilD = 1, dilH = 1, dilW = 1;

    const int64_t outD = deconv_out(x.size(2), weight.size(2), padD, strideD, dilD, outpadD);
    const int64_t outH = deconv_out(x.size(3), weight.size(3), padH, strideH, dilH, outpadH);
    const int64_t outW = deconv_out(x.size(4), weight.size(4), padW, strideW, dilW, outpadW);

    TORCH_CHECK(outD == 18 && outH == 36 && outW == 70,
                "specialized kernel expects output spatial (18,36,70)");

    auto y = at::empty({x.size(0), weight.size(1), outD, outH, outW}, x.options());
    EXEC_NPU_CMD(aclnnConvTransposed3dAsymmetricInputAsymmetricKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed3d_asymmetric_input_asymmetric_kernel_custom",
           &conv_transposed3d_asymmetric_input_asymmetric_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed3d_asymmetric_input_asymmetric_kernel_custom",
          &conv_transposed3d_asymmetric_input_asymmetric_kernel_custom_impl_npu,
          "conv_transposed3d_asymmetric_input_asymmetric_kernel_custom (NPU)");
}
