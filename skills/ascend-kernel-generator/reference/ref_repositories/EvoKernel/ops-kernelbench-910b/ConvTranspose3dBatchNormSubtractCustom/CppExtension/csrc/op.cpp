
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_1d_param(const at::Tensor& t, const char* name, int64_t expected)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.numel() == expected, name, " must have length ", expected);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static inline int64_t convt_out_dim(int64_t in, int64_t stride, int64_t pad, int64_t k, int64_t dil, int64_t out_pad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

at::Tensor conv_transpose3d_batch_norm_subtract_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias)
{
    TORCH_CHECK(x.defined(), "x must be defined");
    TORCH_CHECK(weight.defined(), "weight must be defined");
    TORCH_CHECK(conv_bias.defined(), "conv_bias must be defined");
    TORCH_CHECK(bn_weight.defined(), "bn_weight must be defined");
    TORCH_CHECK(bn_bias.defined(), "bn_bias must be defined");

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(conv_bias.device().type() == c10::DeviceType::PrivateUse1, "conv_bias must be on NPU");
    TORCH_CHECK(bn_weight.device().type() == c10::DeviceType::PrivateUse1, "bn_weight must be on NPU");
    TORCH_CHECK(bn_bias.device().type() == c10::DeviceType::PrivateUse1, "bn_bias must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(conv_bias.scalar_type() == at::kFloat, "conv_bias must be float32");
    TORCH_CHECK(bn_weight.scalar_type() == at::kFloat, "bn_weight must be float32");
    TORCH_CHECK(bn_bias.scalar_type() == at::kFloat, "bn_bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(bn_weight.is_contiguous(), "bn_weight must be contiguous");
    TORCH_CHECK(bn_bias.is_contiguous(), "bn_bias must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cin,Cout,K,K,K] (PyTorch ConvTranspose3d layout)");

    // Specialized contract
    TORCH_CHECK(x.size(1) == 16, "x.size(1) must be 16");
    TORCH_CHECK(x.size(2) == 16 && x.size(3) == 32 && x.size(4) == 32,
                "x spatial must be [16,32,32]");

    TORCH_CHECK(weight.size(0) == 16 && weight.size(1) == 32 &&
                weight.size(2) == 3 && weight.size(3) == 3 && weight.size(4) == 3,
                "weight must be [16,32,3,3,3]");

    check_1d_param(conv_bias, "conv_bias", 32);
    check_1d_param(bn_weight, "bn_weight", 32);
    check_1d_param(bn_bias, "bn_bias", 32);

    constexpr int64_t STR = 2, PAD = 1, DIL = 1, OUT_PAD = 0, K = 3;
    const int64_t Dout = convt_out_dim(x.size(2), STR, PAD, K, DIL, OUT_PAD);
    const int64_t Hout = convt_out_dim(x.size(3), STR, PAD, K, DIL, OUT_PAD);
    const int64_t Wout = convt_out_dim(x.size(4), STR, PAD, K, DIL, OUT_PAD);
    TORCH_CHECK(Dout == 31 && Hout == 63 && Wout == 63,
                "unexpected output shape for specialized params (expects 31x63x63)");

    auto y = at::empty({x.size(0), 32, Dout, Hout, Wout}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConvTranspose3dBatchNormSubtractCustom,
                 x, weight, conv_bias, bn_weight, bn_bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose3d_batch_norm_subtract_custom",
           &conv_transpose3d_batch_norm_subtract_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_batch_norm_subtract_custom",
          &conv_transpose3d_batch_norm_subtract_custom_impl_npu,
          "conv_transpose3d_batch_norm_subtract_custom(x, weight, conv_bias, bn_weight, bn_bias) -> "
          "fused ConvTranspose3d+BatchNorm3d(training stats)+SpatialMeanSubtract (NPU, specialized)");
}
