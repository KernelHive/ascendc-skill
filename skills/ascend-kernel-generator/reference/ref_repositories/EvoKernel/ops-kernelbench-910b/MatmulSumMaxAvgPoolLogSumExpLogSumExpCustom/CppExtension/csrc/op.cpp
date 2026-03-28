
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU (PrivateUse1)");
    TORCH_CHECK(w.device().is_privateuseone(), "w must be on NPU (PrivateUse1)");
    TORCH_CHECK(b.device().is_privateuseone(), "b must be on NPU (PrivateUse1)");

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [B,K]");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [O,K] (Linear weight layout [out,in])");
    TORCH_CHECK(b.dim() == 1, "b must be 1D [O]");

    const int64_t B = x.size(0);
    const int64_t K = x.size(1);
    const int64_t O = w.size(0);

    TORCH_CHECK(w.size(1) == K, "w second dim must equal x second dim (K)");
    TORCH_CHECK(b.size(0) == O, "b size must equal w first dim (O)");

    // Benchmark/model specialization
    TORCH_CHECK(B == 1024, "custom op specialized for batch_size=1024");
    TORCH_CHECK(K == 8192 && O == 8192, "custom op specialized for in_features=8192 and out_features=8192");

    // Output shape must match PyTorch chain: (B,1)
    at::Tensor y = at::empty({B, 1}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMatmulSumMaxAvgPoolLogSumExpLogSumExpCustom, x, w, b, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom",
           &matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom",
          &matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom_impl_npu,
          "matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom (NPU)");
}
