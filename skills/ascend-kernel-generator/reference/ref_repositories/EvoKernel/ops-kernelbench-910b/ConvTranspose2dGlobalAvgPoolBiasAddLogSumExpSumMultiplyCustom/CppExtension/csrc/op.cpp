
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t convt_out_dim(int64_t in, int64_t stride, int64_t pad, int64_t k, int64_t dil, int64_t out_pad) {
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

// Host precompute: sumw[Cin,Cout] = sum_{kh,kw} weight[Cin,Cout,kh,kw]
static at::Tensor precompute_sumw_cpu(const at::Tensor& weight_cpu_contig) {
    TORCH_CHECK(weight_cpu_contig.device().is_cpu(), "weight_cpu_contig must be CPU");
    TORCH_CHECK(weight_cpu_contig.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(weight_cpu_contig.is_contiguous(), "weight must be contiguous on CPU");
    TORCH_CHECK(weight_cpu_contig.dim() == 4, "weight must be 4D (Cin,Cout,Kh,Kw)");
    TORCH_CHECK(weight_cpu_contig.size(2) == 3 && weight_cpu_contig.size(3) == 3, "only 3x3 supported");

    const int64_t Cin  = weight_cpu_contig.size(0);
    const int64_t Cout = weight_cpu_contig.size(1);

    at::Tensor sumw = at::empty({Cin, Cout}, weight_cpu_contig.options());
    const float* wptr = weight_cpu_contig.data_ptr<float>();
    float* sptr = sumw.data_ptr<float>();

    // Layout: (((ci*Cout + co)*3 + kh)*3 + kw)
    for (int64_t ci = 0; ci < Cin; ++ci) {
        for (int64_t co = 0; co < Cout; ++co) {
            const int64_t base = ((ci * Cout + co) * 3) * 3;
            float s = 0.0f;
            s += wptr[base + 0];
            s += wptr[base + 1];
            s += wptr[base + 2];
            s += wptr[base + 3];
            s += wptr[base + 4];
            s += wptr[base + 5];
            s += wptr[base + 6];
            s += wptr[base + 7];
            s += wptr[base + 8];
            sptr[ci * Cout + co] = s;
        }
    }
    return sumw;
}

at::Tensor conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& conv_bias_opt,
    const at::Tensor& bias) // (Cout,1,1) float32
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (Cin,Cout,Kh,Kw)");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D (Cout,1,1)");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must be (Cout,1,1)");

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

    constexpr int64_t STR = 1;
    constexpr int64_t PAD = 0;
    constexpr int64_t DIL = 1;
    constexpr int64_t OUT_PAD = 0;

    TORCH_CHECK(N == 16, "custom op specialized for batch_size=16");
    TORCH_CHECK(Cin == 64 && Cout == 128, "custom op specialized for Cin=64, Cout=128");
    TORCH_CHECK(Hin == 512 && Win == 512, "custom op specialized for H=W=512");
    TORCH_CHECK(bias.sizes().equals(at::IntArrayRef({Cout, 1, 1})), "bias must be (Cout,1,1)");

    const int64_t Hout = convt_out_dim(Hin, STR, PAD, Kh, DIL, OUT_PAD);
    const int64_t Wout = convt_out_dim(Win, STR, PAD, Kw, DIL, OUT_PAD);
    TORCH_CHECK(Hout == 514 && Wout == 514, "unexpected convT output shape for specialized params");

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

    // Precompute sumw on CPU then copy once to NPU (removes 9-tap sum from device hot loop).
    at::Tensor weight_cpu = weight.to(at::kCPU).contiguous();
    at::Tensor sumw_cpu = precompute_sumw_cpu(weight_cpu);
    at::Tensor sumw_npu = sumw_cpu.to(x.device(), /*non_blocking=*/false).contiguous();

    at::Tensor y = at::empty({N, 1}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustom, x, sumw_npu, conv_bias, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom",
           &conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom",
          &conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom_impl_npu,
          "conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom (NPU)");
}
