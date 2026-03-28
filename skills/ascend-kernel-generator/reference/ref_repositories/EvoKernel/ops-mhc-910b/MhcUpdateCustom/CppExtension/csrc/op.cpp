
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor mhc_update_custom_impl_npu(const at::Tensor& x_stream,
                                     const at::Tensor& h_post,
                                     const at::Tensor& h_res,
                                     const at::Tensor& y)
{
    // Preserve original dtype-cast semantics at Python level; kernel is float32-only.
    at::Tensor x32  = (x_stream.scalar_type() == at::kFloat) ? x_stream : x_stream.to(at::kFloat);
    at::Tensor hp32 = (h_post.scalar_type()  == at::kFloat) ? h_post  : h_post.to(at::kFloat);
    at::Tensor hr32 = (h_res.scalar_type()   == at::kFloat) ? h_res   : h_res.to(at::kFloat);
    at::Tensor y32  = (y.scalar_type()       == at::kFloat) ? y       : y.to(at::kFloat);

    x32 = x32.contiguous();
    hp32 = hp32.contiguous();
    hr32 = hr32.contiguous();
    y32 = y32.contiguous();

    TORCH_CHECK(x32.dim() == 4, "x_stream must be 4D (B,T,J,C)");
    TORCH_CHECK(hp32.dim() == 3, "h_post must be 3D (B,T,I)");
    TORCH_CHECK(hr32.dim() == 4, "h_res must be 4D (B,T,I,J)");
    TORCH_CHECK(y32.dim() == 3, "y must be 3D (B,T,C)");

    const int64_t B = x32.size(0);
    const int64_t T = x32.size(1);
    const int64_t J = x32.size(2);
    const int64_t C = x32.size(3);

    TORCH_CHECK(y32.size(0) == B && y32.size(1) == T && y32.size(2) == C, "y shape mismatch");
    TORCH_CHECK(hr32.size(0) == B && hr32.size(1) == T && hr32.size(3) == J, "h_res shape mismatch");

    const int64_t I = hr32.size(2);
    TORCH_CHECK(hp32.size(0) == B && hp32.size(1) == T && hp32.size(2) == I, "h_post shape mismatch");
    TORCH_CHECK(B > 0 && T > 0 && I > 0 && J > 0 && C > 0, "Invalid empty shape");

    at::Tensor out = at::empty({B, T, I, C}, x32.options());
    EXEC_NPU_CMD(aclnnMhcUpdateCustom, x32, hp32, hr32, y32, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_update_custom", &mhc_update_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_update_custom", &mhc_update_custom_impl_npu,
          "mhc_update_custom: fused (btij*btjc + bti*btc) -> btic (AscendC, ping-pong packed panels for I=J=4)");
}
