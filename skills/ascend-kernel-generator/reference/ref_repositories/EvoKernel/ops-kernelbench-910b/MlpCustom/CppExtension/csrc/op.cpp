
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t expected_w_elems(int64_t inSize, int64_t h1, int64_t h2, int64_t outSize) {
    return h1 * inSize + h2 * h1 + outSize * h2;
}
static inline int64_t expected_b_elems(int64_t h1, int64_t h2, int64_t outSize) {
    return h1 + h2 + outSize;
}

at::Tensor mlp_custom_impl_npu(const at::Tensor& x,
                              const at::Tensor& w_packed,
                              const at::Tensor& b_packed)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(w_packed.device().type() == c10::DeviceType::PrivateUse1, "w_packed must be on NPU");
    TORCH_CHECK(b_packed.device().type() == c10::DeviceType::PrivateUse1, "b_packed must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w_packed.scalar_type() == at::kFloat, "w_packed must be float32");
    TORCH_CHECK(b_packed.scalar_type() == at::kFloat, "b_packed must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w_packed.is_contiguous(), "w_packed must be contiguous");
    TORCH_CHECK(b_packed.is_contiguous(), "b_packed must be contiguous");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, in]");
    TORCH_CHECK(w_packed.dim() == 1, "w_packed must be 1D packed tensor");
    TORCH_CHECK(b_packed.dim() == 1, "b_packed must be 1D packed tensor");

    // Fixed-shape contract for this custom op (must match host tiling contract).
    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 16384,
                "x must have shape [128,16384] for mlp_custom");

    const int64_t inSize = 16384;
    const int64_t h1 = 16384;
    const int64_t h2 = 16384;
    const int64_t outSize = 8192;

    TORCH_CHECK(w_packed.numel() == expected_w_elems(inSize, h1, h2, outSize),
                "w_packed numel mismatch for mlp_custom");
    TORCH_CHECK(b_packed.numel() == expected_b_elems(h1, h2, outSize),
                "b_packed numel mismatch for mlp_custom");

    auto y = at::empty({128, 8192}, x.options());
    EXEC_NPU_CMD(aclnnMlpCustom, x, w_packed, b_packed, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mlp_custom", &mlp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mlp_custom", &mlp_custom_impl_npu, "mlp_custom(x, w_packed, b_packed) -> y (NPU)");
}
