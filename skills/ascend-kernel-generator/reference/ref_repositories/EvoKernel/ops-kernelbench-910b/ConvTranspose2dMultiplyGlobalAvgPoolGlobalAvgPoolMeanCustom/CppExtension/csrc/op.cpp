
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convt_out_dim(int64_t in, int64_t stride, int64_t pad, int64_t k, int64_t dil, int64_t out_pad) {
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

at::Tensor conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cin,Cout,Kh,Kw) for ConvTranspose2d");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    const int64_t wCin = weight.size(0);
    const int64_t Cout = weight.size(1);
    const int64_t Kh   = weight.size(2);
    const int64_t Kw   = weight.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    constexpr int64_t STR = 2;
    constexpr int64_t PAD = 1;
    constexpr int64_t OUT_PAD = 1;
    constexpr int64_t DIL = 1;

    TORCH_CHECK(N == 16, "custom op specialized for batch_size=16");
    TORCH_CHECK(Cin == 64 && Cout == 128, "custom op specialized for Cin=64,Cout=128");
    TORCH_CHECK(Hin == 128 && Win == 128, "custom op specialized for input H=W=128");

    const int64_t Hout = convt_out_dim(Hin, STR, PAD, Kh, DIL, OUT_PAD);
    const int64_t Wout = convt_out_dim(Win, STR, PAD, Kw, DIL, OUT_PAD);
    TORCH_CHECK(Hout == 256 && Wout == 256, "unexpected convT output shape for specialized params");

    at::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == Cout, "bias must be 1D with shape [Cout]");
    } else {
        bias = at::zeros({Cout}, x.options().dtype(at::kFloat));
    }

    at::Tensor y = at::empty({N, Cout, 1, 1}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom, x, weight, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom",
           &conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom",
          &conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom_impl_npu,
          "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom (NPU)");
}
