
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

static void check_inputs(const at::Tensor& x,
                         const at::Tensor& clusters,
                         const at::Tensor& clusters2,
                         const at::Tensor& bn_weight,
                         const at::Tensor& bn_bias,
                         const at::Tensor& bn_mean,
                         const at::Tensor& bn_var)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    TORCH_CHECK(clusters.device().type() == c10::DeviceType::PrivateUse1, "clusters must be on NPU");
    TORCH_CHECK(clusters2.device().type() == c10::DeviceType::PrivateUse1, "clusters2 must be on NPU");
    TORCH_CHECK(bn_weight.device().type() == c10::DeviceType::PrivateUse1, "bn_weight must be on NPU");
    TORCH_CHECK(bn_bias.device().type() == c10::DeviceType::PrivateUse1, "bn_bias must be on NPU");
    TORCH_CHECK(bn_mean.device().type() == c10::DeviceType::PrivateUse1, "bn_mean must be on NPU");
    TORCH_CHECK(bn_var.device().type() == c10::DeviceType::PrivateUse1, "bn_var must be on NPU");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(clusters.scalar_type() == at::kFloat, "clusters must be float32");
    TORCH_CHECK(clusters2.scalar_type() == at::kFloat, "clusters2 must be float32");
    TORCH_CHECK(bn_weight.scalar_type() == at::kFloat, "bn_weight must be float32");
    TORCH_CHECK(bn_bias.scalar_type() == at::kFloat, "bn_bias must be float32");
    TORCH_CHECK(bn_mean.scalar_type() == at::kFloat, "bn_mean must be float32");
    TORCH_CHECK(bn_var.scalar_type() == at::kFloat, "bn_var must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(clusters.is_contiguous(), "clusters must be contiguous");
    TORCH_CHECK(clusters2.is_contiguous(), "clusters2 must be contiguous");
    TORCH_CHECK(bn_weight.is_contiguous(), "bn_weight must be contiguous");
    TORCH_CHECK(bn_bias.is_contiguous(), "bn_bias must be contiguous");
    TORCH_CHECK(bn_mean.is_contiguous(), "bn_mean must be contiguous");
    TORCH_CHECK(bn_var.is_contiguous(), "bn_var must be contiguous");

    TORCH_CHECK(x.dim() == 3, "x must be [B,N,D]");
    TORCH_CHECK(clusters.dim() == 2, "clusters must be [D,K]");
    TORCH_CHECK(clusters2.dim() == 3, "clusters2 must be [1,D,K]");
    TORCH_CHECK(bn_weight.dim() == 1 && bn_bias.dim() == 1 && bn_mean.dim() == 1 && bn_var.dim() == 1,
                "bn_* must be 1D [K]");

    TORCH_CHECK(x.size(0) == 2048 && x.size(1) == 100 && x.size(2) == 512,
                "specialized kernel expects x: [2048,100,512]");
    TORCH_CHECK(clusters.size(0) == 512 && clusters.size(1) == 32,
                "specialized kernel expects clusters: [512,32]");
    TORCH_CHECK(clusters2.size(0) == 1 && clusters2.size(1) == 512 && clusters2.size(2) == 32,
                "specialized kernel expects clusters2: [1,512,32]");
    TORCH_CHECK(bn_weight.size(0) == 32 && bn_bias.size(0) == 32 && bn_mean.size(0) == 32 && bn_var.size(0) == 32,
                "specialized kernel expects bn_*: [32]");
}

at::Tensor net_vlad_no_ghost_clusters_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& clusters,
    const at::Tensor& clusters2,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& bn_mean,
    const at::Tensor& bn_var)
{
    check_inputs(x, clusters, clusters2, bn_weight, bn_bias, bn_mean, bn_var);
    at::Tensor y = at::empty({2048, 512 * 32}, x.options());

    EXEC_NPU_CMD(aclnnNetVladNoGhostClustersCustom,
                 x, clusters, clusters2,
                 bn_weight, bn_bias, bn_mean, bn_var,
                 y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("net_vlad_no_ghost_clusters_custom",
           &net_vlad_no_ghost_clusters_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("net_vlad_no_ghost_clusters_custom",
          &net_vlad_no_ghost_clusters_custom_impl_npu,
          "net_vlad_no_ghost_clusters_custom (NPU)");
}
