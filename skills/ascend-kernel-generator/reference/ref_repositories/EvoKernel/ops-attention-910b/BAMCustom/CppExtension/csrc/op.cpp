
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor bam_impl_npu(const at::Tensor& x,
                        const at::Tensor& channel_map,
                        const at::Tensor& spatial_map)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "bam_custom: x must be on NPU");
    TORCH_CHECK(channel_map.device().type() == c10::DeviceType::PrivateUse1, "bam_custom: channel_map must be on NPU");
    TORCH_CHECK(spatial_map.device().type() == c10::DeviceType::PrivateUse1, "bam_custom: spatial_map must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "bam_custom: only float32 supported");
    TORCH_CHECK(channel_map.scalar_type() == at::kFloat, "bam_custom: channel_map must be float32");
    TORCH_CHECK(spatial_map.scalar_type() == at::kFloat, "bam_custom: spatial_map must be float32");

    TORCH_CHECK(x.dim() >= 1, "bam_custom: x must have rank >= 1");
    TORCH_CHECK(channel_map.sizes() == x.sizes(), "bam_custom: channel_map must have same shape as x");
    TORCH_CHECK(spatial_map.sizes() == x.sizes(), "bam_custom: spatial_map must have same shape as x");

    TORCH_CHECK(x.is_contiguous(), "bam_custom: x must be contiguous");
    TORCH_CHECK(channel_map.is_contiguous(), "bam_custom: channel_map must be contiguous");
    TORCH_CHECK(spatial_map.is_contiguous(), "bam_custom: spatial_map must be contiguous");

    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnBAMCustom, x, channel_map, spatial_map, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("bam_custom", &bam_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bam_custom", &bam_impl_npu,
          "BAM fused tail on NPU: y = x * (1 + sigmoid(channel_map + spatial_map)) (float32)");
}
