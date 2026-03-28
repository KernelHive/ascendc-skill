
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_min_subtract_custom_impl_npu(const at::Tensor& x,
                                               const at::Tensor& w,
                                               const at::Tensor& b,
                                               const at::Tensor& c)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "w must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "b must be on NPU");
    TORCH_CHECK(c.device().type() == c10::DeviceType::PrivateUse1, "c must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(c.scalar_type() == at::kFloat, "c must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [N]");
    TORCH_CHECK((c.dim() == 0) || (c.dim() == 1 && c.numel() == 1), "c must be scalar (0D or [1])");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "c must be contiguous");

    // Fixed specialized contract enforced for kernel/tiling simplicity.
    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 16384,
                "x shape must be [128,16384] for matmul_min_subtract_custom");
    TORCH_CHECK(w.size(0) == 16384 && w.size(1) == 16384,
                "w shape must be [16384,16384] (out,in) for matmul_min_subtract_custom");
    TORCH_CHECK(b.size(0) == 16384, "b shape must be [16384] for matmul_min_subtract_custom");

    auto y = at::empty({128, 16384}, x.options());
    EXEC_NPU_CMD(aclnnMatmulMinSubtractCustom, x, w, b, c, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_min_subtract_custom", &matmul_min_subtract_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_min_subtract_custom", &matmul_min_subtract_custom_impl_npu,
          "matmul_min_subtract_custom (NPU)");
}
