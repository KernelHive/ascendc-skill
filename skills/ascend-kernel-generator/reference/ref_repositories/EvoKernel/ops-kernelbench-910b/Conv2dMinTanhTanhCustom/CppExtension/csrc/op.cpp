
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t out_floor(int64_t in, int64_t k, int64_t s, int64_t p, int64_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

static inline void check_4d_nchw_f32_contig_npu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous NCHW");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D (N,C,H,W)");
}

static inline void check_4d_oihw_f32_contig_npu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous [Cout,Cin,Kh,Kw]");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D (Cout,Cin,Kh,Kw)");
}

at::Tensor conv2d_min_tanh_tanh_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt)
{
    check_4d_nchw_f32_contig_npu(x, "x");
    check_4d_oihw_f32_contig_npu(weight, "weight");

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t wCin = weight.size(1);
    const int64_t Kh   = weight.size(2);
    const int64_t Kw   = weight.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin mismatch");

    // Specialization guardrails (must match tiling/kernel/model)
    TORCH_CHECK(N == 128, "custom op specialized for batch_size=128");
    TORCH_CHECK(Cin == 16 && Cout == 64, "custom op specialized for Cin=16, Cout=64");
    TORCH_CHECK(Hin == 256 && Win == 256, "custom op specialized for H=W=256");
    TORCH_CHECK(Kh == 3 && Kw == 3, "custom op specialized for kernel_size=3");

    // Fixed conv hyperparams: stride=1,pad=0,dil=1
    constexpr int64_t STR = 1, PAD = 0, DIL = 1;
    const int64_t Ho = out_floor(Hin, Kh, STR, PAD, DIL);
    const int64_t Wo = out_floor(Win, Kw, STR, PAD, DIL);
    TORCH_CHECK(Ho == 254 && Wo == 254, "unexpected output shape for specialized params");

    // Always pass defined bias to match kernel signature.
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

    // reduce-min keepdim => channel=1
    at::Tensor y = at::empty({N, 1, Ho, Wo}, x.options());
    EXEC_NPU_CMD(aclnnConv2dMinTanhTanhCustom, x, weight, conv_bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_min_tanh_tanh_custom",
           &conv2d_min_tanh_tanh_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_min_tanh_tanh_custom",
          &conv2d_min_tanh_tanh_custom_impl_npu,
          "conv2d_min_tanh_tanh_custom (NPU)");
}
