
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor stream_mix_custom_impl_npu(const at::Tensor& x_stream,
                                     const at::Tensor& h_res)
{
    at::Tensor x = x_stream;
    at::Tensor h = h_res;

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x_stream must be on NPU");
    TORCH_CHECK(h.device().type() == c10::DeviceType::PrivateUse1, "h_res must be on NPU");

    if (h.scalar_type() != x.scalar_type()) {
        h = h.to(x.scalar_type());
    }

    // Kernel is float32-only.
    if (x.scalar_type() != at::kFloat) x = x.to(at::kFloat);
    if (h.scalar_type() != at::kFloat) h = h.to(at::kFloat);

    x = x.contiguous();
    h = h.contiguous();

    TORCH_CHECK(x.dim() == 4, "x_stream must be 4D (B,T,N,C)");
    TORCH_CHECK(h.dim() == 4, "h_res must be 4D (B,T,N,N)");

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t N = x.size(2);
    const int64_t C = x.size(3);

    TORCH_CHECK(B > 0 && T > 0 && N > 0 && C > 0, "Invalid empty shape");
    TORCH_CHECK(h.size(0) == B && h.size(1) == T, "h_res B,T mismatch");
    TORCH_CHECK(h.size(2) == N && h.size(3) == N, "h_res must be (B,T,N,N) matching x_stream N");

    at::Tensor out = at::empty({B, T, N, C}, x.options());

    EXEC_NPU_CMD(aclnnStreamMixCustom, x, h, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("stream_mix_custom", &stream_mix_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream_mix_custom", &stream_mix_custom_impl_npu,
          "stream_mix_custom: (btij @ btjc) -> btic (AscendC)");
}
