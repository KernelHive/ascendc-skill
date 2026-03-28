
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include <cmath>
#include "pytorch_npu_helper.hpp"

static inline int64_t infer_n_from_outcols(int64_t outCols)
{
    const double v = std::sqrt((double)outCols + 1.0);
    if (v < 2.0) return 0;
    const int64_t n = (int64_t)std::floor(v + 1e-9) - 1;
    return n;
}

static inline at::Tensor to_f32_contig_npu(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU");
    at::Tensor o = t;
    if (!o.is_contiguous()) o = o.contiguous();
    if (o.scalar_type() != at::kFloat) o = o.to(at::kFloat);
    return o;
}

static inline at::Tensor to_f32_scalar1_npu(const at::Tensor& t, const char* name)
{
    at::Tensor o = to_f32_contig_npu(t, name);
    TORCH_CHECK(o.numel() == 1, name, " must have exactly 1 element");
    return o.reshape({1}).contiguous();
}

std::vector<at::Tensor> fused_mhc_kernels_custom_impl_npu(
    const at::Tensor& x,          // (B,L,D)
    const at::Tensor& phi,        // (D,outCols) where outCols=n*n+2n
    const at::Tensor& bias,       // (outCols,) or (1,outCols)
    const at::Tensor& scale,      // (D,)
    const at::Tensor& alpha_pre,  // scalar
    const at::Tensor& alpha_post, // scalar
    const at::Tensor& alpha_res,  // scalar
    int64_t iters,
    double eps_rms)
{
    at::Tensor x_c = to_f32_contig_npu(x, "x");
    at::Tensor phi_c = to_f32_contig_npu(phi, "phi");
    at::Tensor bias_c = to_f32_contig_npu(bias, "bias").reshape({-1});
    at::Tensor scale_c = to_f32_contig_npu(scale, "scale").reshape({-1});

    TORCH_CHECK(x_c.dim() == 3, "x must be (B,L,D)");
    TORCH_CHECK(phi_c.dim() == 2, "phi must be (D,outCols)");
    TORCH_CHECK(scale_c.dim() == 1, "scale must be (D,)");

    const int64_t B = x_c.size(0);
    const int64_t L = x_c.size(1);
    const int64_t D = x_c.size(2);
    TORCH_CHECK(B > 0 && L > 0 && D > 0, "invalid B/L/D");

    TORCH_CHECK(phi_c.size(0) == D, "phi first dim must equal D");
    const int64_t outCols = phi_c.size(1);
    TORCH_CHECK(outCols > 0, "phi second dim must be > 0");

    TORCH_CHECK(scale_c.numel() == D, "scale must have D elements");

    const int64_t n = infer_n_from_outcols(outCols);
    TORCH_CHECK(n > 0 && (n * n + 2 * n) == outCols, "phi second dim must be n*n + 2*n");
    TORCH_CHECK(bias_c.numel() == outCols, "bias must have outCols elements");

    at::Tensor ap = to_f32_scalar1_npu(alpha_pre, "alpha_pre");
    at::Tensor ao = to_f32_scalar1_npu(alpha_post, "alpha_post");
    at::Tensor ar = to_f32_scalar1_npu(alpha_res, "alpha_res");

    at::Tensor it_t = at::empty({1}, x_c.options().dtype(at::kInt));
    it_t.fill_(static_cast<int32_t>(iters));

    at::Tensor eps_t = at::empty({1}, x_c.options().dtype(at::kFloat));
    eps_t.fill_(static_cast<float>(eps_rms));

    const float invD = 1.0f / static_cast<float>(D);
    at::Tensor invD_t = at::empty({1}, x_c.options().dtype(at::kFloat));
    invD_t.fill_(invD);

    auto opts = x_c.options().dtype(at::kFloat);
    at::Tensor h_pre  = at::empty({B, L, n}, opts);
    at::Tensor h_post = at::empty({B, L, n}, opts);
    at::Tensor h_res  = at::empty({B, L, n, n}, opts);

    EXEC_NPU_CMD(aclnnFusedMhcKernelsCustom,
                 x_c, phi_c, bias_c, scale_c,
                 ap, ao, ar,
                 it_t, eps_t, invD_t,
                 h_pre, h_post, h_res);

    return {h_pre, h_post, h_res};
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("fused_mhc_kernels_custom", &fused_mhc_kernels_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mhc_kernels_custom", &fused_mhc_kernels_custom_impl_npu,
          "fused_mhc_kernels_custom (NPU): RMSNorm(scale)+projection+bias+alpha+sigmoid+sinkhorn");
}
