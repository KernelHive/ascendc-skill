
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

static inline void check_npu_f32_contig_dim(const at::Tensor& t, const char* name, int64_t dim) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU (PrivateUse1)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == dim, name, " must be ", dim, "D");
}

at::Tensor lstm_cn_custom_impl_npu(const at::Tensor& x,
                                  const at::Tensor& h0,
                                  const at::Tensor& c0,
                                  const at::Tensor& w_ih,
                                  const at::Tensor& w_hh,
                                  const at::Tensor& b_ih,
                                  const at::Tensor& b_hh)
{
    check_npu_f32_contig_dim(x, "x", 3);
    check_npu_f32_contig_dim(h0, "h0", 3);
    check_npu_f32_contig_dim(c0, "c0", 3);
    check_npu_f32_contig_dim(w_ih, "w_ih", 2);
    check_npu_f32_contig_dim(w_hh, "w_hh", 2);
    check_npu_f32_contig_dim(b_ih, "b_ih", 1);
    check_npu_f32_contig_dim(b_hh, "b_hh", 1);

    constexpr int64_t B = 10;
    constexpr int64_t S = 512;
    constexpr int64_t I = 128;
    constexpr int64_t H = 256;
    constexpr int64_t L = 6;
    constexpr int64_t ROWS = L * 4 * H; // 6144

    TORCH_CHECK(x.sizes() == c10::IntArrayRef({B, S, I}),
                "x must be [10,512,128] (batch_first=True)");
    TORCH_CHECK(h0.sizes() == c10::IntArrayRef({L, B, H}),
                "h0 must be [6,10,256]");
    TORCH_CHECK(c0.sizes() == c10::IntArrayRef({L, B, H}),
                "c0 must be [6,10,256]");

    TORCH_CHECK(w_ih.sizes() == c10::IntArrayRef({ROWS, H}),
                "w_ih must be [6144,256] (layer0 weight_ih padded to H columns)");
    TORCH_CHECK(w_hh.sizes() == c10::IntArrayRef({ROWS, H}),
                "w_hh must be [6144,256]");
    TORCH_CHECK(b_ih.numel() == ROWS, "b_ih must have 6144 elements");
    TORCH_CHECK(b_hh.numel() == ROWS, "b_hh must have 6144 elements");

    auto c_n = at::empty({L, B, H}, x.options());
    EXEC_NPU_CMD(aclnnLstmCnCustom, x, h0, c0, w_ih, w_hh, b_ih, b_hh, c_n);
    return c_n;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("lstm_cn_custom", &lstm_cn_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lstm_cn_custom", &lstm_cn_custom_impl_npu, "lstm_cn_custom(x,h0,c0,w_ih,w_hh,b_ih,b_hh)->c_n (NPU, float32)");
}
