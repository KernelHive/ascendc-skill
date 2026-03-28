
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

static at::Tensor conv2d_scaling_min_custom_impl_npu(
    const at::Tensor& x,       // [N,Cin,H,W] float32 NPU
    const at::Tensor& weight,  // [Cout,Cin,Kh,Kw] float32 NPU
    const c10::optional<at::Tensor>& bias_opt, // [Cout] float32 NPU (optional)
    const at::Tensor& scale)   // scalar float32 (CPU or NPU)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(scale.device().is_privateuseone() || scale.device().is_cpu(),
                "scale must be CPU or NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(scale.numel() == 1, "scale must be a 1-element tensor");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cout,Cin,Kh,Kw)");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t H   = x.size(2);
    const int64_t W   = x.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kh   = weight.size(2);
    const int64_t Kw   = weight.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");

    TORCH_CHECK(H >= Kh && W >= Kw, "input H/W must be >= kernel H/W");
    const int64_t Hout = H - Kh + 1;
    const int64_t Wout = W - Kw + 1;

    TORCH_CHECK(N == 64, "custom op specialized for batch_size=64");
    TORCH_CHECK(Cin == 64 && Cout == 128, "custom op specialized for Cin=64,Cout=128");
    TORCH_CHECK(H == 256 && W == 256, "custom op specialized for H=W=256");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");
    TORCH_CHECK(Hout == 254 && Wout == 254, "unexpected output shape");

    const float scl = (scale.device().is_cpu()
                           ? scale.contiguous().item<float>()
                           : scale.cpu().contiguous().item<float>());
    TORCH_CHECK(scl == 2.0f, "custom op specialized for scale_factor=2.0");

    at::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == Cout, "bias must be 1D [Cout]");
    } else {
        bias = at::zeros({Cout}, x.options());
    }

    at::Tensor y = at::empty({N, 1, Hout, Wout}, x.options());
    EXEC_NPU_CMD(aclnnConv2dScalingMinCustom, x, weight, bias, scale, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_scaling_min_custom", &conv2d_scaling_min_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_scaling_min_custom",
          &conv2d_scaling_min_custom_impl_npu,
          "conv2d_scaling_min_custom (NPU)");
}
