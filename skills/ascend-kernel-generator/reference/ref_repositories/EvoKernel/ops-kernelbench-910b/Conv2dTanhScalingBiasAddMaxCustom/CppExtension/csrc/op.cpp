
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

static at::Tensor conv2d_tanh_scaling_bias_add_max_custom_impl_npu(
    const at::Tensor& x,                            // [N,Cin,H,W] float32 NPU
    const at::Tensor& weight,                       // [Cout,Cin,Kh,Kw] float32 NPU
    const c10::optional<at::Tensor>& conv_bias_opt, // [Cout] float32 NPU (optional)
    const at::Tensor& post_bias,                    // [Cout,1,1] or [Cout] float32 NPU
    const at::Tensor& scaling_factor)               // scalar tensor (CPU/NPU), must be 2.0
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(post_bias.device().is_privateuseone(), "post_bias must be on NPU (PrivateUse1)");
    TORCH_CHECK(scaling_factor.device().is_privateuseone() || scaling_factor.device().is_cpu(),
                "scaling_factor must be CPU or NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(post_bias.scalar_type() == at::kFloat, "post_bias must be float32");
    TORCH_CHECK(scaling_factor.scalar_type() == at::kFloat, "scaling_factor must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");
    TORCH_CHECK(scaling_factor.numel() == 1, "scaling_factor must be a 1-element tensor");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cout,Cin,Kh,Kw)");
    TORCH_CHECK(post_bias.dim() == 3 || post_bias.dim() == 1, "post_bias must be [Cout,1,1] or [Cout]");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t H   = x.size(2);
    const int64_t W   = x.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kh   = weight.size(2);
    const int64_t Kw   = weight.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");
    TORCH_CHECK(post_bias.numel() == Cout, "post_bias must have Cout elements");

    TORCH_CHECK(H >= Kh && W >= Kw, "input H/W must be >= kernel H/W");
    const int64_t Hout = H - Kh + 1;
    const int64_t Wout = W - Kw + 1;

    constexpr int64_t poolK = 4;
    constexpr int64_t poolS = 4;
    TORCH_CHECK(Hout >= poolK && Wout >= poolK, "conv output too small for pool");
    const int64_t PHout = (Hout - poolK) / poolS + 1;
    const int64_t PWout = (Wout - poolK) / poolS + 1;

    TORCH_CHECK(N == 128, "custom op specialized for batch_size=128");
    TORCH_CHECK(Cin == 8 && Cout == 64, "custom op specialized for Cin=8,Cout=64");
    TORCH_CHECK(H == 256 && W == 256, "custom op specialized for H=W=256");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");
    TORCH_CHECK(Hout == 254 && Wout == 254, "unexpected conv output shape");
    TORCH_CHECK(PHout == 63 && PWout == 63, "unexpected pool output shape");

    const float scl = (scaling_factor.device().is_cpu()
                           ? scaling_factor.contiguous().item<float>()
                           : scaling_factor.cpu().contiguous().item<float>());
    TORCH_CHECK(scl == 2.0f, "custom op specialized for scaling_factor=2.0");

    at::Tensor conv_bias;
    if (conv_bias_opt.has_value() && conv_bias_opt.value().defined()) {
        conv_bias = conv_bias_opt.value();
        TORCH_CHECK(conv_bias.device().is_privateuseone(), "conv_bias must be on NPU (PrivateUse1)");
        TORCH_CHECK(conv_bias.scalar_type() == at::kFloat, "conv_bias must be float32");
        TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
        TORCH_CHECK(conv_bias.dim() == 1 && conv_bias.size(0) == Cout, "conv_bias must be 1D [Cout]");
    } else {
        conv_bias = at::zeros({Cout}, x.options());
    }

    // Flatten post_bias to [Cout] once
    at::Tensor pb_flat = post_bias.view({Cout}).contiguous();

    at::Tensor sf_npu = scaling_factor;
    if (sf_npu.device().is_cpu()) sf_npu = sf_npu.to(x.device());
    sf_npu = sf_npu.contiguous().view({1});

    at::Tensor y = at::empty({N, Cout, PHout, PWout}, x.options());
    EXEC_NPU_CMD(aclnnConv2dTanhScalingBiasAddMaxCustom, x, weight, conv_bias, pb_flat, sf_npu, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_tanh_scaling_bias_add_max_custom",
           &conv2d_tanh_scaling_bias_add_max_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_tanh_scaling_bias_add_max_custom",
          &conv2d_tanh_scaling_bias_add_max_custom_impl_npu,
          "conv2d_tanh_scaling_bias_add_max_custom (NPU)");
}
