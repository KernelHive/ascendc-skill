
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline void check_npu_f32_contig(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor matmul_scaling_residual_add_custom_impl_npu(const at::Tensor& x,
                                                       const at::Tensor& w,
                                                       const at::Tensor& b,
                                                       const at::Tensor& scaling)
{
    check_npu_f32_contig(x, "x");
    check_npu_f32_contig(w, "w");
    check_npu_f32_contig(b, "b");
    check_npu_f32_contig(scaling, "scaling");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [N]");
    TORCH_CHECK(scaling.dim() == 1 && scaling.numel() == 1, "scaling must be shape [1]");

    // Fixed specialized contract (matches tiling/kernel).
    TORCH_CHECK(x.size(0) == 16384 && x.size(1) == 4096,
                "x shape must be [16384,4096] for matmul_scaling_residual_add_custom");
    TORCH_CHECK(w.size(0) == 4096 && w.size(1) == 4096,
                "w shape must be [4096,4096] (out,in) for matmul_scaling_residual_add_custom");
    TORCH_CHECK(b.size(0) == 4096,
                "b shape must be [4096] for matmul_scaling_residual_add_custom");

    auto y = at::empty({16384, 4096}, x.options());
    EXEC_NPU_CMD(aclnnMatmulScalingResidualAddCustom, x, w, b, scaling, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_scaling_residual_add_custom",
           &matmul_scaling_residual_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_scaling_residual_add_custom",
          &matmul_scaling_residual_add_custom_impl_npu,
          "matmul_scaling_residual_add_custom (NPU)");
}
