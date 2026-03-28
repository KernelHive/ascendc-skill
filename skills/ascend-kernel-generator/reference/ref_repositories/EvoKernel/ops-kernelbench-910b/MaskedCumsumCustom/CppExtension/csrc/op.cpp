
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor masked_cumsum_custom_impl_npu(const at::Tensor& x, const at::Tensor& mask, int64_t dim)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU");
    TORCH_CHECK(mask.device().type() == at::kPrivateUse1, "mask must be on NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte, "mask must be bool or uint8");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have same shape");
    TORCH_CHECK(x.dim() >= 1, "x must have rank >= 1");

    at::Tensor x_c = x.contiguous();

    const int64_t rank = x_c.dim();
    int64_t dim_norm = dim;
    if (dim_norm < 0) dim_norm += rank;
    TORCH_CHECK(dim_norm >= 0 && dim_norm < rank, "dim out of range");
    TORCH_CHECK(dim_norm == rank - 1, "masked_cumsum_custom supports only dim == last dimension");

    // Contract: mask is uint8 on device (0/1).
    at::Tensor m_u8;
    if (mask.scalar_type() == at::kBool) {
        // One-time conversion outside kernel to eliminate bool handling/type punning in device code.
        m_u8 = mask.to(at::kByte).contiguous();
    } else {
        m_u8 = mask.contiguous();
    }

    at::Tensor y = at::empty_like(x_c);
    EXEC_NPU_CMD(aclnnMaskedCumsumCustom, x_c, m_u8, dim_norm, y);
    return y;
}

TORCH_LIBRARY(myops, m) {
    m.def("masked_cumsum_custom(Tensor x, Tensor mask, int dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("masked_cumsum_custom", &masked_cumsum_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_cumsum_custom", &masked_cumsum_custom_impl_npu,
          "masked_cumsum_custom(x, mask, dim) -> masked cumsum on last dim (NPU)");
}
