
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static void check_half_contig_npu(const at::Tensor& t, const char* name, int64_t dim)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be float16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == dim, name, " must be ", dim, "D");
}

at::Tensor gru_custom_impl_npu(const at::Tensor& x,
                              const at::Tensor& h0,
                              const at::Tensor& w_ih,
                              const at::Tensor& w_hh,
                              const at::Tensor& b_ih,
                              const at::Tensor& b_hh)
{
    check_half_contig_npu(x, "x", 3);
    check_half_contig_npu(h0, "h0", 3);
    check_half_contig_npu(w_ih, "w_ih", 2);
    check_half_contig_npu(w_hh, "w_hh", 2);
    check_half_contig_npu(b_ih, "b_ih", 1);
    check_half_contig_npu(b_hh, "b_hh", 1);

    constexpr int64_t T = 512;
    constexpr int64_t B = 10;
    constexpr int64_t I = 128;
    constexpr int64_t H = 256;
    constexpr int64_t L = 6;

    TORCH_CHECK(x.sizes() == c10::IntArrayRef({T, B, I}),
                "x must be [512,10,128] (batch_first=False)");
    TORCH_CHECK(h0.sizes() == c10::IntArrayRef({L, B, H}),
                "h0 must be [6,10,256]");

    TORCH_CHECK(w_ih.sizes() == c10::IntArrayRef({L * 3 * H, H}),
                "w_ih must be packed [4608,256] (layer0 padded to H)");
    TORCH_CHECK(w_hh.sizes() == c10::IntArrayRef({L * 3 * H, H}),
                "w_hh must be [4608,256]");
    TORCH_CHECK(b_ih.numel() == L * 3 * H, "b_ih must have 4608 elements");
    TORCH_CHECK(b_hh.numel() == L * 3 * H, "b_hh must have 4608 elements");

    at::Tensor y = at::empty({T, B, H}, x.options());
    EXEC_NPU_CMD(aclnnGruCustom, x, h0, w_ih, w_hh, b_ih, b_hh, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gru_custom", &gru_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gru_custom", &gru_custom_impl_npu, "gru_custom (NPU)");
}
