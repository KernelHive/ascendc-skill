
#include <torch/library.h>
#include <torch/extension.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor mhc_post_block_custom_impl_npu(
    const at::Tensor& x,               // (N,H) bf16 (external)
    const at::Tensor& residual,        // (N,S,H) bf16 (external)
    const at::Tensor& post_layer_mix,  // (N,S,1) f32
    const at::Tensor& comb_res_mix,    // (N,S,S) f32
    int64_t hidden_size,
    int64_t hc_mult)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(residual.device().is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(post_layer_mix.device().is_privateuseone(), "post_layer_mix must be on NPU");
    TORCH_CHECK(comb_res_mix.device().is_privateuseone(), "comb_res_mix must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(post_layer_mix.scalar_type() == at::kFloat, "post_layer_mix must be float32");
    TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be (N,H)");
    TORCH_CHECK(residual.dim() == 3, "residual must be (N,S,H)");
    TORCH_CHECK(post_layer_mix.dim() == 3, "post_layer_mix must be (N,S,1)");
    TORCH_CHECK(comb_res_mix.dim() == 3, "comb_res_mix must be (N,S,S)");

    const int64_t N = x.size(0);
    const int64_t H = x.size(1);
    TORCH_CHECK(H == hidden_size, "hidden_size must match x.size(1)");
    TORCH_CHECK(residual.size(0) == N && residual.size(2) == H, "residual shape mismatch (N,S,H)");

    const int64_t S = residual.size(1);
    TORCH_CHECK(S == hc_mult, "hc_mult must match residual.size(1)");
    TORCH_CHECK(post_layer_mix.size(0) == N && post_layer_mix.size(1) == S && post_layer_mix.size(2) == 1,
                "post_layer_mix must be (N,S,1)");
    TORCH_CHECK(comb_res_mix.size(0) == N && comb_res_mix.size(1) == S && comb_res_mix.size(2) == S,
                "comb_res_mix must be (N,S,S)");

    TORCH_CHECK(hc_mult == 4, "mhc_post_block_custom currently supports hc_mult==4 only");

    // Kernel is fp16-based internally; cast bfloat16 inputs to fp16 once here.
    at::Tensor x_fp16 = (x.scalar_type() == at::kHalf) ? x.contiguous() : x.to(at::kHalf).contiguous();
    at::Tensor r_fp16 = (residual.scalar_type() == at::kHalf) ? residual.contiguous() : residual.to(at::kHalf).contiguous();
    at::Tensor post_c = post_layer_mix.contiguous();
    at::Tensor comb_c = comb_res_mix.contiguous();

    auto make_i32_scalar = [&](int64_t v) -> at::Tensor {
        at::Tensor t = at::empty({1}, x_fp16.options().dtype(at::kInt));
        t.fill_(static_cast<int32_t>(v));
        return t;
    };

    at::Tensor hs_t = make_i32_scalar(hidden_size);
    at::Tensor hm_t = make_i32_scalar(hc_mult);

    at::Tensor out_fp16 = at::empty({N, hc_mult, hidden_size}, x_fp16.options().dtype(at::kHalf));

    EXEC_NPU_CMD(aclnnMhcPostBlockCustom,
                 x_fp16, r_fp16, post_c, comb_c,
                 hs_t, hm_t,
                 out_fp16);

    // Convert back to bf16 external contract.
    return out_fp16.to(at::kBFloat16);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_post_block_custom", &mhc_post_block_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_post_block_custom", &mhc_post_block_custom_impl_npu,
          "mhc_post_block_custom: fused DeepSeek mHC post-block (AscendC)");
}
