
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& residual,
                         const at::Tensor& fn,
                         const at::Tensor& hc_scale,
                         const at::Tensor& hc_base) {
    TORCH_CHECK(residual.device().type() == c10::DeviceType::PrivateUse1, "residual must be on NPU");
    TORCH_CHECK(fn.device().type() == c10::DeviceType::PrivateUse1, "fn must be on NPU");
    TORCH_CHECK(hc_scale.device().type() == c10::DeviceType::PrivateUse1, "hc_scale must be on NPU");
    TORCH_CHECK(hc_base.device().type() == c10::DeviceType::PrivateUse1, "hc_base must be on NPU");

    TORCH_CHECK(residual.scalar_type() == at::kFloat, "residual must be float32");
    TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
    TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
    TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");

    TORCH_CHECK(residual.dim() == 3, "residual must have shape (N,hc,H)");
    TORCH_CHECK(fn.dim() == 2, "fn must have shape (hc3, hc*H)");
    TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have shape (3,)");
    TORCH_CHECK(hc_base.dim() == 1, "hc_base must have shape (hc3,)");

    const int64_t N = residual.size(0);
    const int64_t hc = residual.size(1);
    const int64_t H = residual.size(2);
    TORCH_CHECK(N > 0 && hc > 0 && H > 0, "invalid residual shape");

    const int64_t dFlat = hc * H;
    const int64_t hc3 = 2 * hc + hc * hc;
    TORCH_CHECK(fn.size(0) == hc3, "fn.size(0) must be hc3=2*hc+hc*hc");
    TORCH_CHECK(fn.size(1) == dFlat, "fn.size(1) must be hc*H");
    TORCH_CHECK(hc_base.numel() == hc3, "hc_base must have hc3 elements");

    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(fn.is_contiguous(), "fn must be contiguous");
    TORCH_CHECK(hc_scale.is_contiguous(), "hc_scale must be contiguous");
    TORCH_CHECK(hc_base.is_contiguous(), "hc_base must be contiguous");
}

std::vector<at::Tensor> mhc_pre_block_custom_impl_npu(const at::Tensor& residual,
                                                     const at::Tensor& fn,
                                                     const at::Tensor& hc_scale,
                                                     const at::Tensor& hc_base) {
    check_inputs(residual, fn, hc_scale, hc_base);

    const auto N = residual.size(0);
    const auto hc = residual.size(1);
    const auto H = residual.size(2);

    auto opts = residual.options().dtype(at::kFloat);
    auto post_mix = at::empty({N, hc, 1}, opts);
    auto comb_mix = at::empty({N, hc, hc}, opts);
    auto layer_input = at::empty({N, H}, opts);

    EXEC_NPU_CMD(aclnnMhcPreBlockCustom, residual, fn, hc_scale, hc_base, post_mix, comb_mix, layer_input);
    return {post_mix, comb_mix, layer_input};
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_pre_block_custom", &mhc_pre_block_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_pre_block_custom", &mhc_pre_block_custom_impl_npu, "mHC pre-block fused op (NPU, fp32 contract)");
}
