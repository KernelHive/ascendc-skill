
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_mish_mish_custom_impl_npu(const at::Tensor& x,
                                           const at::Tensor& weight,
                                           const at::Tensor& bias)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(weight.device().is_privateuseone(), "weight must be on NPU (PrivateUse1)");
    TORCH_CHECK(bias.device().is_privateuseone(), "bias must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N,K] (Linear weight [out,in])");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [N]");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous (call .contiguous() in model)");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);
    const int64_t wK = weight.size(1);

    TORCH_CHECK(wK == K, "weight second dim must equal x second dim (K)");
    TORCH_CHECK(bias.size(0) == N, "bias size must equal N");

    TORCH_CHECK(M == 1024, "custom op specialized for batch_size(M)=1024");
    TORCH_CHECK(K == 8192, "custom op specialized for in_features(K)=8192");
    TORCH_CHECK(N == 8192, "custom op specialized for out_features(N)=8192");

    at::Tensor y = at::empty({M, N}, x.options());
    EXEC_NPU_CMD(aclnnMatmulMishMishCustom, x, weight, bias, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_mish_mish_custom", &matmul_mish_mish_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_mish_mish_custom", &matmul_mish_mish_custom_impl_npu,
          "matmul_mish_mish_custom (NPU)");
}
