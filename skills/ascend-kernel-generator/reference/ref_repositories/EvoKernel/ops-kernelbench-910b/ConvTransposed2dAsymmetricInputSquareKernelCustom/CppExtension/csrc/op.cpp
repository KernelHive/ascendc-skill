
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t deconv_out_1d(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil, int64_t outpad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + outpad + 1;
}

static inline void check_4d_nchw_float_contig_npu(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D NCHW");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (NCHW)");
}

at::Tensor conv_transposed2d_asymmetric_input_square_kernel_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_4d_nchw_float_contig_npu(x, "x");
    check_4d_nchw_float_contig_npu(weight, "weight");

    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [Cin, Cout, Kh, Kw] (groups=1)");

    // Specialization contract (must match kernel constants exactly).
    TORCH_CHECK(x.size(0) == 8,    "specialized kernel expects N=8");
    TORCH_CHECK(x.size(1) == 32,   "specialized kernel expects Cin=32");
    TORCH_CHECK(x.size(2) == 512,  "specialized kernel expects Hin=512");
    TORCH_CHECK(x.size(3) == 1024, "specialized kernel expects Win=1024");

    TORCH_CHECK(weight.size(0) == 32, "specialized kernel expects weight Cin=32");
    TORCH_CHECK(weight.size(1) == 32, "specialized kernel expects weight Cout=32");
    TORCH_CHECK(weight.size(2) == 3,  "specialized kernel expects Kh=3");
    TORCH_CHECK(weight.size(3) == 3,  "specialized kernel expects Kw=3");
    TORCH_CHECK(weight.size(0) == x.size(1), "weight Cin must equal x Cin");

    const int64_t strideH = 1, strideW = 1;
    const int64_t padH = 0, padW = 0;
    const int64_t outpadH = 0, outpadW = 0;
    const int64_t dilH = 1, dilW = 1;

    const int64_t outH = deconv_out_1d(x.size(2), weight.size(2), padH, strideH, dilH, outpadH);
    const int64_t outW = deconv_out_1d(x.size(3), weight.size(3), padW, strideW, dilW, outpadW);

    TORCH_CHECK(outH == 514 && outW == 1026, "specialized kernel expects output spatial (514,1026)");

    auto y = at::empty({x.size(0), weight.size(1), outH, outW}, x.options());

    EXEC_NPU_CMD(aclnnConvTransposed2dAsymmetricInputSquareKernelCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed2d_asymmetric_input_square_kernel_custom",
           &conv_transposed2d_asymmetric_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed2d_asymmetric_input_square_kernel_custom",
          &conv_transposed2d_asymmetric_input_square_kernel_custom_impl_npu,
          "conv_transposed2d_asymmetric_input_square_kernel_custom (NPU)");
}
