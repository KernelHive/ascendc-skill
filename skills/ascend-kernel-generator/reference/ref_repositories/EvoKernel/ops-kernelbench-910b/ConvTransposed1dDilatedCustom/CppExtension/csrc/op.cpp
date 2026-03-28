
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t deconv_out_1d(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil, int64_t outpad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + outpad + 1;
}

static inline void check_3d_ncl_float_contig_npu(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D NCL");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (NCL)");
}

at::Tensor conv_transposed1d_dilated_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_3d_ncl_float_contig_npu(x, "x");
    check_3d_ncl_float_contig_npu(weight, "weight");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [Cin, Cout, K] (groups=1)");

    // Specialization contract (must match kernel constants exactly).
    TORCH_CHECK(x.size(0) == 32,     "specialized kernel expects N=32");
    TORCH_CHECK(x.size(1) == 32,     "specialized kernel expects Cin=32");
    TORCH_CHECK(x.size(2) == 131072, "specialized kernel expects Lin=131072");

    TORCH_CHECK(weight.size(0) == 32, "specialized kernel expects weight Cin=32");
    TORCH_CHECK(weight.size(1) == 64, "specialized kernel expects weight Cout=64");
    TORCH_CHECK(weight.size(2) == 5,  "specialized kernel expects K=5");
    TORCH_CHECK(weight.size(0) == x.size(1), "weight Cin must equal x Cin");

    const int64_t stride = 1;
    const int64_t pad = 0;
    const int64_t dilation = 3;
    const int64_t outpad = 0;

    const int64_t lout = deconv_out_1d(x.size(2), weight.size(2), pad, stride, dilation, outpad);
    TORCH_CHECK(lout == 131084, "specialized kernel expects Lout=131084");

    auto y = at::empty({x.size(0), weight.size(1), lout}, x.options());

    EXEC_NPU_CMD(aclnnConvTransposed1dDilatedCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transposed1d_dilated_custom",
           &conv_transposed1d_dilated_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transposed1d_dilated_custom",
          &conv_transposed1d_dilated_custom_impl_npu,
          "conv_transposed1d_dilated_custom (NPU)");
}
