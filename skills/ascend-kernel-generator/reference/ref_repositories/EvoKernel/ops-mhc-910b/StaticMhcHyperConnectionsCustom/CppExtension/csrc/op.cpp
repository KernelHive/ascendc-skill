
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include <cmath>
#include "pytorch_npu_helper.hpp"

at::Tensor static_mhc_hyper_connections_custom_impl_npu(
    const at::Tensor& residuals,      // (B,S,D)
    const at::Tensor& h_res_logits,   // (S,S)
    const at::Tensor& h_pre_logits,   // (S)
    const at::Tensor& h_post_logits,  // (S)
    const at::Tensor& branch_weight,  // (D,D)
    int64_t sinkhorn_iters,
    double sinkhorn_tau)
{
    TORCH_CHECK(residuals.device().is_privateuseone(), "residuals must be on NPU");
    TORCH_CHECK(residuals.dim() == 3, "residuals must be (B,S,D)");
    TORCH_CHECK(h_res_logits.dim() == 2, "h_res_logits must be (S,S)");
    TORCH_CHECK(h_pre_logits.dim() == 1, "h_pre_logits must be (S)");
    TORCH_CHECK(h_post_logits.dim() == 1, "h_post_logits must be (S)");
    TORCH_CHECK(branch_weight.dim() == 2, "branch_weight must be (D,D)");
    TORCH_CHECK(sinkhorn_tau > 0.0, "sinkhorn_tau must be > 0");

    // float32-only contract for kernel
    at::Tensor r  = residuals.contiguous();     if (r.scalar_type()  != at::kFloat) r  = r.to(at::kFloat);
    at::Tensor hr = h_res_logits.contiguous();  if (hr.scalar_type() != at::kFloat) hr = hr.to(at::kFloat);
    at::Tensor hp = h_pre_logits.contiguous();  if (hp.scalar_type() != at::kFloat) hp = hp.to(at::kFloat);
    at::Tensor ho = h_post_logits.contiguous(); if (ho.scalar_type() != at::kFloat) ho = ho.to(at::kFloat);
    at::Tensor w  = branch_weight.contiguous(); if (w.scalar_type()  != at::kFloat) w  = w.to(at::kFloat);

    const int64_t B = r.size(0), S = r.size(1), D = r.size(2);
    TORCH_CHECK(B > 0 && S > 0 && D > 0, "invalid B/S/D");
    TORCH_CHECK(S <= 16, "static_mhc_hyper_connections_custom supports S<=16");
    TORCH_CHECK(hr.size(0) == S && hr.size(1) == S, "h_res_logits must be (S,S)");
    TORCH_CHECK(hp.size(0) == S, "h_pre_logits must be (S)");
    TORCH_CHECK(ho.size(0) == S, "h_post_logits must be (S)");
    TORCH_CHECK(w.size(0) == D && w.size(1) == D, "branch_weight must be (D,D)");

    at::Tensor it_t = at::empty({1}, r.options().dtype(at::kInt));
    it_t.fill_(static_cast<int32_t>(sinkhorn_iters));

    at::Tensor tau_t = at::empty({1}, r.options().dtype(at::kFloat));
    tau_t.fill_(static_cast<float>(sinkhorn_tau));

    at::Tensor log_s_t = at::empty({1}, r.options().dtype(at::kFloat));
    log_s_t.fill_(static_cast<float>(std::log(static_cast<double>(S))));

    at::Tensor out = at::empty({B, S, D}, r.options().dtype(at::kFloat));

    // IMPORTANT: keep naming consistent with OpDef: StaticMhcHyperConnectionsCustom => aclnnStaticMhcHyperConnectionsCustom
    EXEC_NPU_CMD(aclnnStaticMhcHyperConnectionsCustom,
                 r, hr, hp, ho, w,
                 it_t, tau_t, log_s_t,
                 out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("static_mhc_hyper_connections_custom", &static_mhc_hyper_connections_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("static_mhc_hyper_connections_custom", &static_mhc_hyper_connections_custom_impl_npu,
          "static_mhc_hyper_connections_custom (NPU AscendC): sinkhorn(h_res)+softmax(h_pre/h_post)+einsums+linear fused; float32; S<=16; add_branch_out_to_residual=True");
}
