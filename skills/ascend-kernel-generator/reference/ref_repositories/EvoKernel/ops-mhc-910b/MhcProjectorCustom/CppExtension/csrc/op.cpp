
#include <torch/library.h>
#include <torch/extension.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> mhc_projector_custom_impl_npu(
    const at::Tensor& x_stream,
    const at::Tensor& phi_pre,
    const at::Tensor& phi_post,
    const at::Tensor& phi_res,
    const at::Tensor& b_pre,
    const at::Tensor& b_post,
    const at::Tensor& b_res,
    const at::Tensor& alpha_pre,
    const at::Tensor& alpha_post,
    const at::Tensor& alpha_res,
    int64_t tmax,
    double rmsnorm_eps)
{
    TORCH_CHECK(x_stream.device().is_privateuseone(), "x_stream must be on NPU");
    TORCH_CHECK(x_stream.dim() == 4, "x_stream must be (B,T,N,C)");

    at::Tensor xs = x_stream.contiguous();
    if (xs.scalar_type() != at::kFloat) xs = xs.to(at::kFloat);

    const int64_t B = xs.size(0), T = xs.size(1), N = xs.size(2), C = xs.size(3);
    TORCH_CHECK(N > 0 && C > 0, "invalid N/C");
    const int64_t F = N * C;
    const int64_t NN = N * N;

    at::Tensor pp = phi_pre.contiguous();  if (pp.scalar_type() != at::kFloat) pp = pp.to(at::kFloat);
    at::Tensor po = phi_post.contiguous(); if (po.scalar_type() != at::kFloat) po = po.to(at::kFloat);
    at::Tensor pr = phi_res.contiguous();  if (pr.scalar_type() != at::kFloat) pr = pr.to(at::kFloat);

    TORCH_CHECK(pp.sizes() == at::IntArrayRef({F, N}), "phi_pre must be (N*C, N)");
    TORCH_CHECK(po.sizes() == at::IntArrayRef({F, N}), "phi_post must be (N*C, N)");
    TORCH_CHECK(pr.sizes() == at::IntArrayRef({F, NN}), "phi_res must be (N*C, N*N)");

    at::Tensor bp = b_pre.contiguous();  if (bp.scalar_type() != at::kFloat) bp = bp.to(at::kFloat);
    at::Tensor bo = b_post.contiguous(); if (bo.scalar_type() != at::kFloat) bo = bo.to(at::kFloat);
    TORCH_CHECK(bp.sizes() == at::IntArrayRef({N}), "b_pre must be (N)");
    TORCH_CHECK(bo.sizes() == at::IntArrayRef({N}), "b_post must be (N)");

    at::Tensor br = b_res;
    if (br.scalar_type() != at::kFloat) br = br.to(at::kFloat);
    TORCH_CHECK(br.numel() == NN, "b_res must have N*N elements");
    br = br.reshape({NN}).contiguous();

    auto to_scalar_f32_1 = [&](const at::Tensor& t, const char* name) -> at::Tensor {
        at::Tensor o = t;
        if (o.scalar_type() != at::kFloat) o = o.to(at::kFloat);
        TORCH_CHECK(o.numel() == 1, name, " must have 1 element");
        return o.reshape({1}).contiguous();
    };

    at::Tensor ap = to_scalar_f32_1(alpha_pre, "alpha_pre");
    at::Tensor ao = to_scalar_f32_1(alpha_post, "alpha_post");
    at::Tensor ar = to_scalar_f32_1(alpha_res, "alpha_res");

    at::Tensor tmax_t = at::empty({1}, xs.options().dtype(at::kInt));
    tmax_t.fill_(static_cast<int32_t>(tmax));
    at::Tensor eps_t = at::empty({1}, xs.options().dtype(at::kFloat));
    eps_t.fill_(static_cast<float>(rmsnorm_eps));

    const float invF = 1.0f / static_cast<float>(F);
    at::Tensor invF_t = at::empty({1}, xs.options().dtype(at::kFloat));
    invF_t.fill_(invF);

    at::Tensor h_pre  = at::empty({B, T, N}, xs.options().dtype(at::kFloat));
    at::Tensor h_post = at::empty({B, T, N}, xs.options().dtype(at::kFloat));
    at::Tensor h_res  = at::empty({B, T, N, N}, xs.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnMhcProjectorCustom,
                 xs, pp, po, pr,
                 bp, bo, br,
                 ap, ao, ar,
                 tmax_t, eps_t, invF_t,
                 h_pre, h_post, h_res);

    return {h_pre, h_post, h_res};
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_projector_custom", &mhc_projector_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_projector_custom", &mhc_projector_custom_impl_npu,
          "mhc_projector_custom: RMSNorm + projections + sigmoid + sinkhorn (NPU AscendC)");
}
