
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

at::Tensor vanilla_rnn_hidden_custom_impl_npu(const at::Tensor& x,
                                             const at::Tensor& h0,
                                             const at::Tensor& w_i2h,
                                             const at::Tensor& b_i2h,
                                             const at::Tensor& w_h2o,
                                             const at::Tensor& b_h2o)
{
    check_npu_f32_contig_dim(x, "x", 3);
    check_npu_f32_contig_dim(h0, "h0", 2);
    check_npu_f32_contig_dim(w_i2h, "w_i2h", 2);
    check_npu_f32_contig_dim(b_i2h, "b_i2h", 1);
    check_npu_f32_contig_dim(w_h2o, "w_h2o", 2);
    check_npu_f32_contig_dim(b_h2o, "b_h2o", 1);

    constexpr int64_t T = 256;
    constexpr int64_t B = 8;
    constexpr int64_t I = 1024;
    constexpr int64_t H = 256;
    constexpr int64_t O = 128;
    constexpr int64_t K = I + H;

    TORCH_CHECK(x.sizes() == c10::IntArrayRef({T, B, I}),
                "x must be [256,8,1024] (seq_len,batch,input)");
    TORCH_CHECK(h0.sizes() == c10::IntArrayRef({B, H}),
                "h0 must be [8,256] (batch,hidden)");

    TORCH_CHECK(w_i2h.sizes() == c10::IntArrayRef({H, K}),
                "w_i2h must be [256,1280] (Linear(H, I+H) weight)");
    TORCH_CHECK(b_i2h.sizes() == c10::IntArrayRef({H}),
                "b_i2h must be [256]");
    TORCH_CHECK(w_h2o.sizes() == c10::IntArrayRef({O, H}),
                "w_h2o must be [128,256]");
    TORCH_CHECK(b_h2o.sizes() == c10::IntArrayRef({O}),
                "b_h2o must be [128]");

    auto y = at::empty({T, B, O}, x.options()); // float32 output
    EXEC_NPU_CMD(aclnnVanillaRnnHiddenCustom, x, h0, w_i2h, b_i2h, w_h2o, b_h2o, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("vanilla_rnn_hidden_custom", &vanilla_rnn_hidden_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vanilla_rnn_hidden_custom", &vanilla_rnn_hidden_custom_impl_npu,
          "vanilla_rnn_hidden_custom(x,h0,w_i2h,b_i2h,w_h2o,b_h2o)->y (NPU, float32)");
}
