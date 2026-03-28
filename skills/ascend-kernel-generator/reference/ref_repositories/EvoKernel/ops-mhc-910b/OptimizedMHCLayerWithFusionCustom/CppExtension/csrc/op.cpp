
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& x,
                         const at::Tensor& phi_params,
                         const at::Tensor& bias_params,
                         const at::Tensor& rms_scale,
                         const at::Tensor& alpha_pre,
                         const at::Tensor& alpha_post,
                         const at::Tensor& alpha_res,
                         const at::Tensor& linear_w) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(phi_params.device().type() == c10::DeviceType::PrivateUse1, "phi_params must be on NPU");
    TORCH_CHECK(bias_params.device().type() == c10::DeviceType::PrivateUse1, "bias_params must be on NPU");
    TORCH_CHECK(rms_scale.device().type() == c10::DeviceType::PrivateUse1, "rms_scale must be on NPU");
    TORCH_CHECK(alpha_pre.device().type() == c10::DeviceType::PrivateUse1, "alpha_pre must be on NPU");
    TORCH_CHECK(alpha_post.device().type() == c10::DeviceType::PrivateUse1, "alpha_post must be on NPU");
    TORCH_CHECK(alpha_res.device().type() == c10::DeviceType::PrivateUse1, "alpha_res must be on NPU");
    TORCH_CHECK(linear_w.device().type() == c10::DeviceType::PrivateUse1, "linear_w must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(phi_params.scalar_type() == at::kFloat, "phi_params must be float32");
    TORCH_CHECK(bias_params.scalar_type() == at::kFloat, "bias_params must be float32");
    TORCH_CHECK(rms_scale.scalar_type() == at::kFloat, "rms_scale must be float32");
    TORCH_CHECK(alpha_pre.scalar_type() == at::kFloat, "alpha_pre must be float32");
    TORCH_CHECK(alpha_post.scalar_type() == at::kFloat, "alpha_post must be float32");
    TORCH_CHECK(alpha_res.scalar_type() == at::kFloat, "alpha_res must be float32");
    TORCH_CHECK(linear_w.scalar_type() == at::kFloat, "linear_w must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(phi_params.is_contiguous(), "phi_params must be contiguous");
    TORCH_CHECK(bias_params.is_contiguous(), "bias_params must be contiguous");
    TORCH_CHECK(rms_scale.is_contiguous(), "rms_scale must be contiguous");
    TORCH_CHECK(alpha_pre.is_contiguous(), "alpha_pre must be contiguous");
    TORCH_CHECK(alpha_post.is_contiguous(), "alpha_post must be contiguous");
    TORCH_CHECK(alpha_res.is_contiguous(), "alpha_res must be contiguous");
    TORCH_CHECK(linear_w.is_contiguous(), "linear_w must be contiguous");

    TORCH_CHECK(x.dim() == 3, "x must have shape (B,S,D)");
    TORCH_CHECK(phi_params.dim() == 2, "phi_params must have shape (SD,mapDim)");
    TORCH_CHECK(bias_params.dim() == 1, "bias_params must be shape (mapDim,)");
    TORCH_CHECK(rms_scale.dim() == 1, "rms_scale must be shape (SD,)");
    TORCH_CHECK(alpha_pre.numel() >= 1, "alpha_pre must have >=1 element");
    TORCH_CHECK(alpha_post.numel() >= 1, "alpha_post must have >=1 element");
    TORCH_CHECK(alpha_res.numel() >= 1, "alpha_res must have >=1 element");

    TORCH_CHECK(linear_w.dim() == 2, "linear_w must be 2D");
    TORCH_CHECK(linear_w.size(0) == linear_w.size(1), "linear_w must be square");

    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t D = x.size(2);
    TORCH_CHECK(B > 0 && S > 0 && D > 0, "invalid x shape");

    TORCH_CHECK(linear_w.size(0) == D, "linear_w must have shape (D,D) matching x.size(2)");

    const int64_t SD = phi_params.size(0);
    const int64_t mapDim = phi_params.size(1);
    TORCH_CHECK(SD > 0 && mapDim > 0, "invalid phi_params shape");
    TORCH_CHECK(bias_params.numel() == mapDim, "bias_params must have mapDim elements");
    TORCH_CHECK(rms_scale.numel() == SD, "rms_scale must have SD elements");

    // Specialization constraints: n=4, mapDim=24, SD=4*D.
    TORCH_CHECK(mapDim == 24, "specialization requires mapDim=24 (n=4)");
    TORCH_CHECK(SD == 4 * D, "specialization requires phi_params.size(0)=SD=4*D (n=4)");
}

at::Tensor optimized_mhc_layer_with_fusion_custom_impl_npu(const at::Tensor& x,
                                                          const at::Tensor& phi_params,
                                                          const at::Tensor& bias_params,
                                                          const at::Tensor& rms_scale,
                                                          const at::Tensor& alpha_pre,
                                                          const at::Tensor& alpha_post,
                                                          const at::Tensor& alpha_res,
                                                          const at::Tensor& linear_w) {
    check_inputs(x, phi_params, bias_params, rms_scale, alpha_pre, alpha_post, alpha_res, linear_w);

    auto y = at::empty_like(x, x.options().dtype(at::kFloat));

    // IMPORTANT: Op name is OptimizedMHCLayerWithFusionCustom (MHC uppercase) -> aclnnOptimizedMHCLayerWithFusionCustom
    EXEC_NPU_CMD(aclnnOptimizedMHCLayerWithFusionCustom,
                 x, phi_params, bias_params, rms_scale, alpha_pre, alpha_post, alpha_res, linear_w, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("optimized_mhc_layer_with_fusion_custom", &optimized_mhc_layer_with_fusion_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimized_mhc_layer_with_fusion_custom",
          &optimized_mhc_layer_with_fusion_custom_impl_npu,
          "Optimized mHC layer full forward (NPU, fp32)");
}
