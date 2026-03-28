
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor stream_write_custom_impl_npu(const at::Tensor& y, const at::Tensor& h_post) {
    at::Tensor h = h_post;
    if (h.scalar_type() != y.scalar_type()) {
        h = h.to(y.scalar_type());
    }
    const auto B = y.size(0);
    const auto T = y.size(1);
    const auto C = y.size(2);
    const auto N = h.size(2);
    at::Tensor out = at::empty({B, T, N, C}, y.options());
    EXEC_NPU_CMD(aclnnStreamWriteCustom, y, h, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("stream_write_custom", &stream_write_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream_write_custom", &stream_write_custom_impl_npu,
          "stream_write_custom: (B,T,C) and (B,T,N) -> (B,T,N,C)");
}
