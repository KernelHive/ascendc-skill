
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

// Pack weight from [16,3,3,3,3] -> [81,16] on NPU for contiguous per-tap loads.
// This is a lightweight transpose+reshape relative to total compute, and enables much faster kernel inner loop.
static at::Tensor pack_weight_81x16_npu(const at::Tensor& w)
{
    // w: [Cout,Cin,Kd,Kh,Kw] = [16,3,3,3,3]
    // permute to [Cin,Kd,Kh,Kw,Cout] then reshape to [81,16]
    return w.permute({1,2,3,4,0}).contiguous().view({81, 16});
}

at::Tensor conv3d_softmax_max_pool_max_pool_custom_impl_npu(const at::Tensor& x,
                                                           const at::Tensor& weight,
                                                           const at::Tensor& bias)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1, "weight must be on NPU");
    TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1, "bias must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [Cout,Cin,Kd,Kh,Kw]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [Cout]");

    TORCH_CHECK(x.size(0) == 128 && x.size(1) == 3 && x.size(2) == 16 && x.size(3) == 32 && x.size(4) == 32,
                "x shape must be [128,3,16,32,32] for conv3d_softmax_max_pool_max_pool_custom");
    TORCH_CHECK(weight.size(0) == 16 && weight.size(1) == 3 &&
                weight.size(2) == 3 && weight.size(3) == 3 && weight.size(4) == 3,
                "weight shape must be [16,3,3,3,3] for conv3d_softmax_max_pool_max_pool_custom");
    TORCH_CHECK(bias.numel() == 16, "bias shape must be [16] for conv3d_softmax_max_pool_max_pool_custom");

    auto weight_packed = pack_weight_81x16_npu(weight);
    TORCH_CHECK(weight_packed.is_contiguous(), "packed weight must be contiguous");

    auto y = at::empty({128, 16, 3, 7, 7}, x.options());
    EXEC_NPU_CMD(aclnnConv3dSoftmaxMaxPoolMaxPoolCustom, x, weight_packed, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_softmax_max_pool_max_pool_custom",
           &conv3d_softmax_max_pool_max_pool_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_softmax_max_pool_max_pool_custom",
          &conv3d_softmax_max_pool_max_pool_custom_impl_npu,
          "conv3d_softmax_max_pool_max_pool_custom (NPU)");
}
