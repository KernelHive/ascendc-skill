
#include <torch/library.h>
#include <torch/extension.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor mhc_block2d_custom_impl_npu(
    const at::Tensor& x,        // (B,64,32,32) bf16/fp16
    const at::Tensor& out,      // (B,64,32,32) bf16/fp16 (post-bn2)
    const at::Tensor& map_w,    // (16,64) fp32
    const at::Tensor& map_bias, // (16) fp32
    int64_t num_streams,
    int64_t sinkhorn_iter,
    double sinkhorn_eps,
    double sinkhorn_temp,
    int64_t in_channels,
    int64_t out_channels,
    int64_t height,
    int64_t width)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(out.device().is_privateuseone(), "out must be on NPU");
    TORCH_CHECK(map_w.device().is_privateuseone(), "map_w must be on NPU");
    TORCH_CHECK(map_bias.device().is_privateuseone(), "map_bias must be on NPU");

    TORCH_CHECK(x.dim() == 4 && out.dim() == 4, "x/out must be (B,C,H,W)");
    const int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");

    TORCH_CHECK(num_streams == 4, "mhc_block2d_custom supports num_streams==4 only");
    TORCH_CHECK(in_channels == 64 && out_channels == 64, "mhc_block2d_custom supports in/out_channels==64 only");
    TORCH_CHECK(C == 64, "x.size(1) must be 64");
    TORCH_CHECK(height == H && width == W, "height/width must match x");
    TORCH_CHECK(H == 32 && W == 32, "mhc_block2d_custom supports 32x32 only");

    TORCH_CHECK(map_w.scalar_type() == at::kFloat, "map_w must be float32");
    TORCH_CHECK(map_bias.scalar_type() == at::kFloat, "map_bias must be float32");
    TORCH_CHECK(map_w.dim() == 2 && map_w.size(0) == 16 && map_w.size(1) == 64, "map_w must be (16,64)");
    TORCH_CHECK(map_bias.numel() == 16, "map_bias must have 16 elements");

    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "x must be bf16 or fp16");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf, "out must be bf16 or fp16");

    if (sinkhorn_iter < 0) sinkhorn_iter = 0;
    if (sinkhorn_iter > 50) sinkhorn_iter = 50;
    if (!(sinkhorn_eps > 0.0)) sinkhorn_eps = 1e-8;
    if (sinkhorn_temp == 0.0) sinkhorn_temp = 1.0;

    // Kernel uses fp16 IO for broad compatibility.
    at::Tensor x_fp16   = (x.scalar_type()   == at::kHalf) ? x.contiguous()   : x.to(at::kHalf).contiguous();
    at::Tensor out_fp16 = (out.scalar_type() == at::kHalf) ? out.contiguous() : out.to(at::kHalf).contiguous();

    at::Tensor mw = map_w.contiguous();
    at::Tensor mb = map_bias.contiguous();

    auto make_i32_scalar = [&](int64_t v) -> at::Tensor {
        at::Tensor t = at::empty({1}, x_fp16.options().dtype(at::kInt));
        t.fill_(static_cast<int32_t>(v));
        return t;
    };
    auto make_f32_scalar = [&](double v) -> at::Tensor {
        at::Tensor t = at::empty({1}, x_fp16.options().dtype(at::kFloat));
        t.fill_(static_cast<float>(v));
        return t;
    };

    at::Tensor ns_t = make_i32_scalar(num_streams);
    at::Tensor it_t = make_i32_scalar(sinkhorn_iter);
    at::Tensor se_t = make_f32_scalar(sinkhorn_eps);
    at::Tensor st_t = make_f32_scalar(sinkhorn_temp);

    at::Tensor ic_t = make_i32_scalar(in_channels);
    at::Tensor oc_t = make_i32_scalar(out_channels);
    at::Tensor h_t  = make_i32_scalar(height);
    at::Tensor w_t  = make_i32_scalar(width);

    at::Tensor y_fp16 = at::empty_like(out_fp16, out_fp16.options().dtype(at::kHalf));

    EXEC_NPU_CMD(aclnnMhcBlock2dCustom,
                 x_fp16, out_fp16,
                 mw, mb,
                 ns_t, it_t, se_t, st_t,
                 ic_t, oc_t, h_t, w_t,
                 y_fp16);

    return (out.scalar_type() == at::kBFloat16) ? y_fp16.to(at::kBFloat16) : y_fp16;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_block2d_custom", &mhc_block2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_block2d_custom", &mhc_block2d_custom_impl_npu,
          "mhc_block2d_custom: fused pool+linear+sinkhorn+stream-mix on NPU (AscendC)");
}
