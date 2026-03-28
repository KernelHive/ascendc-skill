
#include <torch/library.h>
#include <torch/extension.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

static at::Tensor to_f32_contig_npu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU");
    at::Tensor o = t.contiguous();
    if (o.scalar_type() != at::kFloat) o = o.to(at::kFloat);
    TORCH_CHECK(o.is_contiguous(), name, " must be contiguous");
    return o;
}

static at::Tensor scalar_i32_npu_like(const at::Tensor& like, int64_t v) {
    at::Tensor t = at::empty({1}, like.options().dtype(at::kInt));
    t.fill_(static_cast<int32_t>(v));
    return t;
}

static at::Tensor scalar_f32_npu_like(const at::Tensor& like, double v) {
    at::Tensor t = at::empty({1}, like.options().dtype(at::kFloat));
    t.fill_(static_cast<float>(v));
    return t;
}

static at::Tensor scalar_f32_1_npu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU");
    TORCH_CHECK(t.numel() == 1, name, " must have 1 element");
    at::Tensor o = t.reshape({1}).contiguous();
    if (o.scalar_type() != at::kFloat) o = o.to(at::kFloat);
    return o;
}

static std::vector<at::Tensor> dummy_mlp_tensors_like(const at::Tensor& like) {
    auto opt = like.options().dtype(at::kFloat);
    at::Tensor w1 = at::zeros({1,1}, opt);
    at::Tensor b1 = at::zeros({1}, opt);
    at::Tensor w2 = at::zeros({1,1}, opt);
    at::Tensor b2 = at::zeros({1}, opt);
    at::Tensor lnw = at::ones({1}, opt);
    at::Tensor lnb = at::zeros({1}, opt);
    at::Tensor lne = at::zeros({1}, opt);
    lne.fill_(1e-5f);
    return {w1,b1,w2,b2,lnw,lnb,lne};
}

at::Tensor mhc_module_custom_impl_npu(
    const at::Tensor& x_streams,   // (B,T,N,C)
    const at::Tensor& rms_scale,   // (F)
    const at::Tensor& w_pre,       // (F,N)
    const at::Tensor& w_post,      // (F,N)
    const at::Tensor& w_res,       // (F,NN)
    const at::Tensor& b_pre,       // (N)
    const at::Tensor& b_post,      // (N)
    const at::Tensor& b_res,       // (NN)
    const at::Tensor& alpha_pre,   // scalar
    const at::Tensor& alpha_post,  // scalar
    const at::Tensor& alpha_res,   // scalar
    int64_t tmax,
    double rms_eps,
    bool use_mlp,
    // Optional MLPResidual parameters (required for correctness when use_mlp=true):
    const c10::optional<at::Tensor>& mlp_w1_opt, // (C,H)
    const c10::optional<at::Tensor>& mlp_b1_opt, // (H)
    const c10::optional<at::Tensor>& mlp_w2_opt, // (H,C)
    const c10::optional<at::Tensor>& mlp_b2_opt, // (C)
    const c10::optional<at::Tensor>& ln_weight_opt, // (C)
    const c10::optional<at::Tensor>& ln_bias_opt,   // (C)
    const c10::optional<double>& ln_eps_opt
) {
    TORCH_CHECK(x_streams.device().is_privateuseone(), "x_streams must be on NPU");
    TORCH_CHECK(x_streams.dim() == 4, "x_streams must be (B,T,N,C)");
    at::Tensor xs = to_f32_contig_npu(x_streams, "x_streams");

    const int64_t B = xs.size(0), T = xs.size(1), N = xs.size(2), C = xs.size(3);
    TORCH_CHECK(B > 0 && T > 0 && N > 0 && C > 0, "invalid x_streams shape");
    const int64_t F = N * C;
    const int64_t NN = N * N;

    at::Tensor sc = to_f32_contig_npu(rms_scale.reshape({F}), "rms_scale");
    TORCH_CHECK(sc.numel() == F, "rms_scale must have F elements");

    at::Tensor wp = to_f32_contig_npu(w_pre, "w_pre");
    at::Tensor wo = to_f32_contig_npu(w_post, "w_post");
    at::Tensor wr = to_f32_contig_npu(w_res, "w_res");
    TORCH_CHECK(wp.sizes() == at::IntArrayRef({F, N}), "w_pre must be (F,N)");
    TORCH_CHECK(wo.sizes() == at::IntArrayRef({F, N}), "w_post must be (F,N)");
    TORCH_CHECK(wr.sizes() == at::IntArrayRef({F, NN}), "w_res must be (F,N*N)");

    at::Tensor bp = to_f32_contig_npu(b_pre.reshape({N}), "b_pre");
    at::Tensor bo = to_f32_contig_npu(b_post.reshape({N}), "b_post");
    at::Tensor br = to_f32_contig_npu(b_res.reshape({NN}), "b_res");

    at::Tensor ap = scalar_f32_1_npu(alpha_pre, "alpha_pre");
    at::Tensor ao = scalar_f32_1_npu(alpha_post, "alpha_post");
    at::Tensor ar = scalar_f32_1_npu(alpha_res, "alpha_res");

    at::Tensor tmax_t = scalar_i32_npu_like(xs, tmax);
    at::Tensor eps_t  = scalar_f32_npu_like(xs, rms_eps);
    const float invF = 1.0f / static_cast<float>(F);
    at::Tensor invF_t = scalar_f32_npu_like(xs, invF);
    at::Tensor use_t  = scalar_i32_npu_like(xs, use_mlp ? 1 : 0);

    at::Tensor mlp_w1, mlp_b1, mlp_w2, mlp_b2, ln_w, ln_b, ln_eps_t;
    if (use_mlp) {
        TORCH_CHECK(mlp_w1_opt.has_value() && mlp_b1_opt.has_value() && mlp_w2_opt.has_value() && mlp_b2_opt.has_value()
                    && ln_weight_opt.has_value() && ln_bias_opt.has_value(),
                    "When use_mlp=True, must pass mlp_w1/mlp_b1/mlp_w2/mlp_b2 and ln_weight/ln_bias");
        const double ln_eps = ln_eps_opt.has_value() ? *ln_eps_opt : 1e-5;
        mlp_w1 = to_f32_contig_npu(*mlp_w1_opt, "mlp_w1");
        mlp_b1 = to_f32_contig_npu(*mlp_b1_opt, "mlp_b1");
        mlp_w2 = to_f32_contig_npu(*mlp_w2_opt, "mlp_w2");
        mlp_b2 = to_f32_contig_npu(*mlp_b2_opt, "mlp_b2");
        ln_w   = to_f32_contig_npu(*ln_weight_opt, "ln_weight");
        ln_b   = to_f32_contig_npu(*ln_bias_opt, "ln_bias");

        TORCH_CHECK(mlp_w1.dim()==2 && mlp_w1.size(0)==C, "mlp_w1 must be (C,H)");
        TORCH_CHECK(mlp_w2.dim()==2 && mlp_w2.size(1)==C, "mlp_w2 must be (H,C)");
        TORCH_CHECK(mlp_w2.size(0)==mlp_w1.size(1), "mlp_w2 first dim must match H");
        TORCH_CHECK(mlp_b1.numel()==mlp_w1.size(1), "mlp_b1 must be (H)");
        TORCH_CHECK(mlp_b2.numel()==C, "mlp_b2 must be (C)");
        TORCH_CHECK(ln_w.numel()==C && ln_b.numel()==C, "ln_weight/ln_bias must be (C)");

        ln_eps_t = scalar_f32_npu_like(xs, ln_eps);
    } else {
        auto d = dummy_mlp_tensors_like(xs);
        mlp_w1 = d[0]; mlp_b1=d[1]; mlp_w2=d[2]; mlp_b2=d[3]; ln_w=d[4]; ln_b=d[5]; ln_eps_t=d[6];
    }

    at::Tensor out = at::empty({B, T, N, C}, xs.options().dtype(at::kFloat));

    // Keep ACLNN symbol casing consistent with op name: MhcModuleCustom -> aclnnMhcModuleCustom
    EXEC_NPU_CMD(aclnnMhcModuleCustom,
                 xs, sc,
                 wp, wo, wr,
                 bp, bo, br,
                 ap, ao, ar,
                 tmax_t, eps_t, invF_t,
                 use_t,
                 mlp_w1, mlp_b1, mlp_w2, mlp_b2,
                 ln_w, ln_b, ln_eps_t,
                 out);

    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_module_custom", &mhc_module_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_module_custom", &mhc_module_custom_impl_npu,
          "mhc_module_custom: fused mHC module (RMSNorm+projections+sinkhorn+einsums+optional MLPResidual) (NPU AscendC)",
          py::arg("x_streams"),
          py::arg("rms_scale"),
          py::arg("w_pre"),
          py::arg("w_post"),
          py::arg("w_res"),
          py::arg("b_pre"),
          py::arg("b_post"),
          py::arg("b_res"),
          py::arg("alpha_pre"),
          py::arg("alpha_post"),
          py::arg("alpha_res"),
          py::arg("tmax"),
          py::arg("rms_eps"),
          py::arg("use_mlp"),
          py::arg("mlp_w1") = py::none(),
          py::arg("mlp_b1") = py::none(),
          py::arg("mlp_w2") = py::none(),
          py::arg("mlp_b2") = py::none(),
          py::arg("ln_weight") = py::none(),
          py::arg("ln_bias") = py::none(),
          py::arg("ln_eps") = py::none());
}
