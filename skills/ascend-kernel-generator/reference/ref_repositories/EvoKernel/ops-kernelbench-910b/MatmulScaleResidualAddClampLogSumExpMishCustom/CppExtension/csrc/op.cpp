
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline void check_npu_f32_contig(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor matmul_scale_residual_add_clamp_log_sum_exp_mish_custom_impl_npu(
    const at::Tensor& x,         // [1024,8192]
    const at::Tensor& weight,    // [8192,8192] (out,in)
    const at::Tensor& bias,      // [8192]
    const at::Tensor& scaling,   // [1]
    const at::Tensor& clamp_min, // [1]
    const at::Tensor& clamp_max  // [1]
)
{
    check_npu_f32_contig(x, "x");
    check_npu_f32_contig(weight, "weight");
    check_npu_f32_contig(bias, "bias");
    check_npu_f32_contig(scaling, "scaling");
    check_npu_f32_contig(clamp_min, "clamp_min");
    check_npu_f32_contig(clamp_max, "clamp_max");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [N]");
    TORCH_CHECK(scaling.dim() == 1 && scaling.numel() == 1, "scaling must be shape [1]");
    TORCH_CHECK(clamp_min.dim() == 1 && clamp_min.numel() == 1, "clamp_min must be shape [1]");
    TORCH_CHECK(clamp_max.dim() == 1 && clamp_max.numel() == 1, "clamp_max must be shape [1]");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);
    const int64_t wK = weight.size(1);

    TORCH_CHECK(wK == K, "weight second dim must equal x second dim (K)");
    TORCH_CHECK(bias.size(0) == N, "bias size must equal N");

    // Specialization guardrails
    TORCH_CHECK(M == 1024, "custom op specialized for batch_size(M)=1024");
    TORCH_CHECK(K == 8192, "custom op specialized for input_size(K)=8192");
    TORCH_CHECK(N == 8192, "custom op specialized for hidden_size(N)=8192");

    // Output is logsumexp over dim=1 keepdim=True -> [M,1]
    at::Tensor y = at::empty({M, 1}, x.options());
    EXEC_NPU_CMD(aclnnMatmulScaleResidualAddClampLogSumExpMishCustom,
                 x, weight, bias, scaling, clamp_min, clamp_max, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_scale_residual_add_clamp_log_sum_exp_mish_custom",
           &matmul_scale_residual_add_clamp_log_sum_exp_mish_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_scale_residual_add_clamp_log_sum_exp_mish_custom",
          &matmul_scale_residual_add_clamp_log_sum_exp_mish_custom_impl_npu,
          "matmul_scale_residual_add_clamp_log_sum_exp_mish_custom (NPU)");
}
