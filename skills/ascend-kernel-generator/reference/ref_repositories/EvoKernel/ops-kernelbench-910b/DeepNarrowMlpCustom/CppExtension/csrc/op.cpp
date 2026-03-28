
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline int64_t expected_w_elems(int64_t inSize, int64_t hiddenSize, int64_t outSize, int64_t numHidden) {
    return hiddenSize * inSize + (numHidden - 1) * hiddenSize * hiddenSize + outSize * hiddenSize;
}
static inline int64_t expected_b_elems(int64_t hiddenSize, int64_t outSize, int64_t numHidden) {
    return numHidden * hiddenSize + outSize;
}

at::Tensor deep_narrow_mlp_custom_impl_npu(const at::Tensor& x,
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

    // Specialized contract for this kernel (matches host tiling contract).
    TORCH_CHECK(x.size(0) == 1024 && x.size(1) == 8192,
                "x must have shape [1024,8192] for deep_narrow_mlp_custom");

    const int64_t hiddenSize = 1024;
    const int64_t numHidden = 16;
    const int64_t outSize = 8192;

    TORCH_CHECK(w_packed.numel() == expected_w_elems(8192, hiddenSize, outSize, numHidden),
                "w_packed numel mismatch for deep_narrow_mlp_custom");
    TORCH_CHECK(b_packed.numel() == expected_b_elems(hiddenSize, outSize, numHidden),
                "b_packed numel mismatch for deep_narrow_mlp_custom");

    auto y = at::empty({1024, 8192}, x.options());
    EXEC_NPU_CMD(aclnnDeepNarrowMlpCustom, x, w_packed, b_packed, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("deep_narrow_mlp_custom", &deep_narrow_mlp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deep_narrow_mlp_custom", &deep_narrow_mlp_custom_impl_npu,
          "deep_narrow_mlp_custom(x, w_packed, b_packed) -> y (NPU)");
}
