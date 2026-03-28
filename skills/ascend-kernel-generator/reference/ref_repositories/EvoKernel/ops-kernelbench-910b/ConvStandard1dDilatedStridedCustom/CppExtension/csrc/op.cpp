
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t conv1d_out_nopad(int64_t Lin, int64_t stride, int64_t dilation, int64_t K)
{
    const int64_t effective = dilation * (K - 1) + 1;
    TORCH_CHECK(Lin >= effective, "Invalid Lin: Lin=", Lin, " effective=", effective);
    return (Lin - effective) / stride + 1;
}

static inline void check_npu_f32_contig_3d(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D (NCL / OCK)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor conv_standard1d_dilated_strided_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight)
{
    check_npu_f32_contig_3d(x, "x");
    check_npu_f32_contig_3d(weight, "weight");

    TORCH_CHECK(x.size(0) == 64, "specialized kernel expects N=64");
    TORCH_CHECK(x.size(1) == 64, "specialized kernel expects Cin=64");
    TORCH_CHECK(x.size(2) == 524280, "specialized kernel expects Lin=524280");

    TORCH_CHECK(weight.size(0) == 128, "specialized kernel expects Cout=128");
    TORCH_CHECK(weight.size(1) == 64, "specialized kernel expects weight.Cin=64");
    TORCH_CHECK(weight.size(2) == 3, "specialized kernel expects K=3");

    constexpr int64_t stride = 3;
    constexpr int64_t dilation = 4;
    const int64_t Lout = conv1d_out_nopad(x.size(2), stride, dilation, weight.size(2));
    TORCH_CHECK(Lout == 174758, "specialized kernel expects Lout=174758");

    auto y = at::empty({64, 128, Lout}, x.options());
    EXEC_NPU_CMD(aclnnConvStandard1dDilatedStridedCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard1d_dilated_strided_custom",
           &conv_standard1d_dilated_strided_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard1d_dilated_strided_custom",
          &conv_standard1d_dilated_strided_custom_impl_npu,
          "conv_standard1d_dilated_strided_custom (NPU)");
}
