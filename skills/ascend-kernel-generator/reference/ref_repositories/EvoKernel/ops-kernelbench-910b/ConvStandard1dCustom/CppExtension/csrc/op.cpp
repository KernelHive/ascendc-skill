
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void check_specialized(const at::Tensor& x, const at::Tensor& weight) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N,C,L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (Cout,Cin,K)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    TORCH_CHECK(x.size(0) == 32 && x.size(1) == 64 && x.size(2) == 131072,
                "Specialized op supports only x=[32,64,131072]");
    TORCH_CHECK(weight.size(0) == 128 && weight.size(1) == 64 && weight.size(2) == 3,
                "Specialized op supports only weight=[128,64,3]");
}

at::Tensor conv_standard1d_custom_impl_npu(const at::Tensor& x,
                                          const at::Tensor& weight)
{
    check_specialized(x, weight);

    constexpr int64_t N = 32;
    constexpr int64_t COUT = 128;
    constexpr int64_t LOUT = 131070;

    auto y = at::empty({N, COUT, LOUT}, x.options());
    EXEC_NPU_CMD(aclnnConvStandard1dCustom, x, weight, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("conv_standard1d_custom(Tensor x, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_standard1d_custom", &conv_standard1d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard1d_custom", &conv_standard1d_custom_impl_npu,
          "conv_standard1d_custom (NPU)");
}
