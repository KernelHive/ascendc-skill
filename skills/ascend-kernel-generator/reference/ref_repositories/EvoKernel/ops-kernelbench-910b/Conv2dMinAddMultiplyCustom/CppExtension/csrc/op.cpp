
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

static at::Tensor conv2d_min_add_multiply_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt,
    const at::Tensor& post_bias,
    const at::Tensor& constant_value,
    const at::Tensor& scaling_factor)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(post_bias.device().is_privateuseone(), "post_bias must be on NPU (PrivateUse1)");
    TORCH_CHECK(constant_value.device().is_privateuseone() || constant_value.device().is_cpu(),
                "constant_value must be CPU or NPU");
    TORCH_CHECK(scaling_factor.device().is_privateuseone() || scaling_factor.device().is_cpu(),
                "scaling_factor must be CPU or NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(post_bias.scalar_type() == at::kFloat, "post_bias must be float32");
    TORCH_CHECK(constant_value.scalar_type() == at::kFloat, "constant_value must be float32");
    TORCH_CHECK(scaling_factor.scalar_type() == at::kFloat, "scaling_factor must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");
    TORCH_CHECK(constant_value.numel() == 1, "constant_value must be a 1-element tensor");
    TORCH_CHECK(scaling_factor.numel() == 1, "scaling_factor must be a 1-element tensor");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cout,Cin,Kh,Kw)");
    TORCH_CHECK(post_bias.dim() == 3, "post_bias must be 3D [Cout,1,1]");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t H   = x.size(2);
    const int64_t W   = x.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kh   = weight.size(2);
    const int64_t Kw   = weight.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");
    TORCH_CHECK(post_bias.size(0) == Cout && post_bias.size(1) == 1 && post_bias.size(2) == 1,
                "post_bias must be [Cout,1,1]");

    TORCH_CHECK(H >= Kh && W >= Kw, "input H/W must be >= kernel H/W");
    const int64_t Hout = H - Kh + 1;
    const int64_t Wout = W - Kw + 1;

    TORCH_CHECK(N == 128, "custom op specialized for batch_size=128");
    TORCH_CHECK(Cin == 64 && Cout == 128, "custom op specialized for Cin=64,Cout=128");
    TORCH_CHECK(H == 128 && W == 128, "custom op specialized for H=W=128");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");
    TORCH_CHECK(Hout == 126 && Wout == 126, "unexpected output shape");

    const float cst = (constant_value.device().is_cpu()
                           ? constant_value.contiguous().item<float>()
                           : constant_value.cpu().contiguous().item<float>());
    const float scl = (scaling_factor.device().is_cpu()
                           ? scaling_factor.contiguous().item<float>()
                           : scaling_factor.cpu().contiguous().item<float>());
    TORCH_CHECK(cst == 0.5f, "custom op specialized for constant_value=0.5");
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

    at::Tensor y = at::empty({N, Cout, Hout, Wout}, x.options());
    EXEC_NPU_CMD(aclnnConv2dMinAddMultiplyCustom, x, weight, conv_bias, post_bias, constant_value, scaling_factor, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_min_add_multiply_custom", &conv2d_min_add_multiply_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_min_add_multiply_custom",
          &conv2d_min_add_multiply_custom_impl_npu,
          "conv2d_min_add_multiply_custom (NPU)");
}
