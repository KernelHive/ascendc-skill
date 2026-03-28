
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor lct_impl_npu(const at::Tensor& x,
                        const at::Tensor& w,
                        const at::Tensor& b,
                        int64_t groups,
                        double eps)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "lct_custom: x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "lct_custom: w must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "lct_custom: b must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "lct_custom: x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "lct_custom: w must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "lct_custom: b must be float32");

    TORCH_CHECK(x.dim() == 4, "lct_custom: x must be [N,C,H,W]");
    TORCH_CHECK(x.is_contiguous(), "lct_custom: x must be contiguous NCHW");
    TORCH_CHECK(x.size(2) > 0 && x.size(3) > 0, "lct_custom: H/W must be positive");

    TORCH_CHECK(w.dim() == 1 && b.dim() == 1, "lct_custom: w and b must be [C]");
    TORCH_CHECK(w.is_contiguous(), "lct_custom: w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "lct_custom: b must be contiguous");

    const auto C = x.size(1);
    TORCH_CHECK(w.size(0) == C, "lct_custom: w.shape[0] must equal C");
    TORCH_CHECK(b.size(0) == C, "lct_custom: b.shape[0] must equal C");

    TORCH_CHECK(groups > 0, "lct_custom: groups must be > 0");
    TORCH_CHECK((C % groups) == 0, "lct_custom: C must be divisible by groups");
    TORCH_CHECK(eps >= 0.0, "lct_custom: eps must be >= 0");

    at::Tensor g_t = at::empty({}, x.options().dtype(at::kInt));
    g_t.fill_(static_cast<int32_t>(groups));

    at::Tensor eps_t = at::empty({}, x.options().dtype(at::kFloat));
    eps_t.fill_(static_cast<float>(eps));

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnLCTCustom, x, w, b, g_t, eps_t, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("lct_custom(Tensor x, Tensor w, Tensor b, int groups, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("lct_custom", &lct_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lct_custom", &lct_impl_npu,
          "LCT fused op (GAP + group-norm over channels + affine + sigmoid + scale) on NPU");
}
