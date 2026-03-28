
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <cstdint>
#include <limits>

static inline void grouped_gemm_custom_check(const at::Tensor& lhs,
                                             const at::Tensor& rhs,
                                             const at::Tensor& m_indices)
{
    TORCH_CHECK(lhs.defined() && rhs.defined() && m_indices.defined(),
                "grouped_gemm_custom: inputs must be defined");

    TORCH_CHECK(lhs.device().type() == c10::DeviceType::PrivateUse1,
                "grouped_gemm_custom: lhs must be on NPU");
    TORCH_CHECK(rhs.device().type() == c10::DeviceType::PrivateUse1,
                "grouped_gemm_custom: rhs must be on NPU");
    TORCH_CHECK(m_indices.device().type() == c10::DeviceType::PrivateUse1,
                "grouped_gemm_custom: m_indices must be on NPU");

    TORCH_CHECK(lhs.scalar_type() == at::kBFloat16,
                "grouped_gemm_custom: lhs must be bfloat16");
    TORCH_CHECK(rhs.scalar_type() == at::kBFloat16,
                "grouped_gemm_custom: rhs must be bfloat16");
    TORCH_CHECK(m_indices.scalar_type() == at::kInt,
                "grouped_gemm_custom: m_indices must be int32");

    TORCH_CHECK(lhs.is_contiguous(),
                "grouped_gemm_custom: lhs must be contiguous");
    TORCH_CHECK(rhs.is_contiguous(),
                "grouped_gemm_custom: rhs must be contiguous");
    TORCH_CHECK(m_indices.is_contiguous(),
                "grouped_gemm_custom: m_indices must be contiguous");

    TORCH_CHECK(lhs.dim() == 2,
                "grouped_gemm_custom: lhs must be 2D [M,K], got dim=", lhs.dim());
    TORCH_CHECK(rhs.dim() == 3,
                "grouped_gemm_custom: rhs must be 3D [G,N,K], got dim=", rhs.dim());
    TORCH_CHECK(m_indices.dim() == 1,
                "grouped_gemm_custom: m_indices must be 1D [M], got dim=", m_indices.dim());

    const int64_t M = lhs.size(0);
    const int64_t K = lhs.size(1);
    const int64_t G = rhs.size(0);
    const int64_t N = rhs.size(1);
    const int64_t K2 = rhs.size(2);

    TORCH_CHECK(M > 0 && K > 0 && N > 0 && G > 0,
                "grouped_gemm_custom: empty dimensions not supported");
    TORCH_CHECK(K2 == K,
                "grouped_gemm_custom: K mismatch: rhs.size(2)=", K2, " vs lhs.size(1)=", K);
    TORCH_CHECK(m_indices.numel() == M,
                "grouped_gemm_custom: m_indices.numel must equal M, got ", m_indices.numel(), " vs ", M);

    TORCH_CHECK(K <= 8192, "grouped_gemm_custom: K too large (max 8192), got ", K);
    TORCH_CHECK(N <= 8192, "grouped_gemm_custom: N too large (max 8192), got ", N);
    TORCH_CHECK(G <= 4096, "grouped_gemm_custom: num_groups too large (max 4096), got ", G);
    TORCH_CHECK(M <= (int64_t)std::numeric_limits<int32_t>::max(),
                "grouped_gemm_custom: M too large for this kernel");
}

at::Tensor grouped_gemm_custom_impl_npu(const at::Tensor& lhs,
                                       const at::Tensor& rhs,
                                       const at::Tensor& m_indices)
{
    grouped_gemm_custom_check(lhs, rhs, m_indices);
    const int64_t M = lhs.size(0);
    const int64_t N = rhs.size(1);
    at::Tensor out = at::empty({M, N}, lhs.options());
    EXEC_NPU_CMD(aclnnGroupedGEMMCustom, lhs, rhs, m_indices, out);
    return out;
}

TORCH_LIBRARY(myops, m) {
    m.def("grouped_gemm_custom(Tensor lhs, Tensor rhs, Tensor m_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("grouped_gemm_custom", &grouped_gemm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm_custom", &grouped_gemm_custom_impl_npu,
          "GroupedGEMMCustom (NPU, bf16): out[i,n] = sum_k lhs[i,k] * rhs[m_indices[i], n, k]");
}
