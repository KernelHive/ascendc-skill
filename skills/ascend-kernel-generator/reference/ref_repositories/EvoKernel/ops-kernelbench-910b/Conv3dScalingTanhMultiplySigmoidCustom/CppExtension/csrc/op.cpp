
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_dim_floor(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil) {
    const int64_t eff = dil * (k - 1) + 1;
    TORCH_CHECK(in + 2 * pad >= eff, "Invalid conv params produce negative output dim");
    return (in + 2 * pad - eff) / stride + 1;
}

// y = sigmoid( tanh( (conv3d(x,w,conv_bias) * scaling_factor) ) * bias )
at::Tensor conv3d_scaling_tanh_multiply_sigmoid_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt,
    const at::Tensor& scaling_factor,
    const at::Tensor& bias)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(scaling_factor.device().is_privateuseone(), "scaling_factor must be on NPU (PrivateUse1)");
    TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(scaling_factor.scalar_type() == at::kFloat, "scaling_factor must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(scaling_factor.is_contiguous(), "scaling_factor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (Cout,Cin/groups,Kd,Kh,Kw)");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Din = x.size(2);
    const int64_t Hin = x.size(3);
    const int64_t Win = x.size(4);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kd   = weight.size(2);
    const int64_t Kh   = weight.size(3);
    const int64_t Kw   = weight.size(4);

    // Specialized conv params
    constexpr int64_t STR = 1;
    constexpr int64_t PAD = 0;
    constexpr int64_t DIL = 1;
    constexpr int64_t GRP = 1;

    TORCH_CHECK(GRP == 1, "custom op specialized for groups=1");
    TORCH_CHECK(wCin == Cin, "weight Cin mismatch for groups=1");
    TORCH_CHECK(Kd == 3 && Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    // Guardrails: benchmark specialization
    TORCH_CHECK(N == 128, "custom op specialized for batch_size=128");
    TORCH_CHECK(Cin == 3 && Cout == 16, "custom op specialized for Cin=3,Cout=16");
    TORCH_CHECK(Din == 16 && Hin == 64 && Win == 64, "custom op specialized for input D/H/W = 16/64/64");

    // scaling_factor and bias expected broadcastable [Cout,1,1,1] (or [Cout])
    TORCH_CHECK(scaling_factor.numel() == Cout, "scaling_factor must have Cout elements (e.g. [Cout,1,1,1])");
    TORCH_CHECK(bias.numel() == Cout, "bias must have Cout elements (e.g. [Cout,1,1,1])");
    TORCH_CHECK((scaling_factor.dim() == 4 || scaling_factor.dim() == 1),
                "scaling_factor must be [Cout,1,1,1] or [Cout]");
    TORCH_CHECK((bias.dim() == 4 || bias.dim() == 1),
                "bias must be [Cout,1,1,1] or [Cout]");

    const int64_t Dout = out_dim_floor(Din, Kd, PAD, STR, DIL);
    const int64_t Hout = out_dim_floor(Hin, Kh, PAD, STR, DIL);
    const int64_t Wout = out_dim_floor(Win, Kw, PAD, STR, DIL);
    TORCH_CHECK(Dout == 14 && Hout == 62 && Wout == 62, "unexpected output shape for specialized params");

    // conv_bias: always materialize defined [Cout]
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

    // Flatten to [Cout] for kernel broadcast
    at::Tensor sf_flat = scaling_factor.view({Cout}).contiguous();
    at::Tensor b_flat  = bias.view({Cout}).contiguous();

    at::Tensor y = at::empty({N, Cout, Dout, Hout, Wout}, x.options());
    EXEC_NPU_CMD(aclnnConv3dScalingTanhMultiplySigmoidCustom, x, weight, conv_bias, sf_flat, b_flat, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_scaling_tanh_multiply_sigmoid_custom",
           &conv3d_scaling_tanh_multiply_sigmoid_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_scaling_tanh_multiply_sigmoid_custom",
          &conv3d_scaling_tanh_multiply_sigmoid_custom_impl_npu,
          "conv3d_scaling_tanh_multiply_sigmoid_custom (NPU)");
}
