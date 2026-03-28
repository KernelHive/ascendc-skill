
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline void check_1d_npu_float_contig(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
static inline void check_2d_npu_float_contig(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor gemm_bias_add_hardtanh_mish_group_norm_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& lin_bias,
    const at::Tensor& bias,
    const at::Tensor& gn_gamma,
    const at::Tensor& gn_beta)
{
    check_2d_npu_float_contig(x, "x");
    check_2d_npu_float_contig(weight, "weight");
    check_1d_npu_float_contig(lin_bias, "lin_bias");
    check_1d_npu_float_contig(bias, "bias");
    check_1d_npu_float_contig(gn_gamma, "gn_gamma");
    check_1d_npu_float_contig(gn_beta, "gn_beta");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);
    const int64_t wK = weight.size(1);

    TORCH_CHECK(wK == K, "weight second dim must equal x second dim (K)");
    TORCH_CHECK(lin_bias.size(0) == N, "lin_bias must be [N]");
    TORCH_CHECK(bias.size(0) == N, "bias must be [N]");
    TORCH_CHECK(gn_gamma.size(0) == N, "gn_gamma must be [N]");
    TORCH_CHECK(gn_beta.size(0) == N, "gn_beta must be [N]");

    TORCH_CHECK(M == 1024, "custom op specialized for batch_size(M)=1024");
    TORCH_CHECK(K == 8192, "custom op specialized for in_features(K)=8192");
    TORCH_CHECK(N == 8192, "custom op specialized for out_features(N)=8192");

    auto y = at::empty({M, N}, x.options());
    EXEC_NPU_CMD(aclnnGemmBiasAddHardtanhMishGroupNormCustom,
                 x, weight, lin_bias, bias, gn_gamma, gn_beta, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_bias_add_hardtanh_mish_group_norm_custom",
           &gemm_bias_add_hardtanh_mish_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_bias_add_hardtanh_mish_group_norm_custom",
          &gemm_bias_add_hardtanh_mish_group_norm_custom_impl_npu,
          "gemm_bias_add_hardtanh_mish_group_norm_custom (NPU)");
}
