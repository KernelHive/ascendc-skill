
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor stream_weighted_sum_custom_impl_npu(const at::Tensor& x_stream,
                                               const at::Tensor& weights)
{
    at::Tensor x = x_stream;
    at::Tensor w = weights;
    if (w.scalar_type() != x.scalar_type()) {
        w = w.to(x.scalar_type());
    }

    // Op spec is float32 only; enforce float32 for both.
    if (x.scalar_type() != at::kFloat) x = x.to(at::kFloat);
    if (w.scalar_type() != at::kFloat) w = w.to(at::kFloat);

    x = x.contiguous();
    w = w.contiguous();

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x_stream must be on NPU");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1, "weights must be on NPU");

    TORCH_CHECK(x.dim() == 4, "x_stream must be 4D (B,T,N,C)");
    TORCH_CHECK(w.dim() == 3, "weights must be 3D (B,T,N)");

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t N = x.size(2);
    const int64_t C = x.size(3);

    TORCH_CHECK(B > 0 && T > 0 && N > 0 && C > 0, "Invalid empty shape");
    TORCH_CHECK(w.size(0) == B && w.size(1) == T && w.size(2) == N,
                "shape mismatch: weights must be (B,T,N) matching x_stream (B,T,N,C)");

    at::Tensor out = at::empty({B, T, C}, x.options());

    EXEC_NPU_CMD(aclnnStreamWeightedSumCustom, x, w, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("stream_weighted_sum_custom", &stream_weighted_sum_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream_weighted_sum_custom", &stream_weighted_sum_custom_impl_npu,
          "stream_weighted_sum_custom: einsum('btn,btnc->btc') fused reduction (AscendC)");
}
