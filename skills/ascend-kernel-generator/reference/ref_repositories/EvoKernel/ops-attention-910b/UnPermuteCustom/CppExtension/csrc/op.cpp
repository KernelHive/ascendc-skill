
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <cstdint>
#include <limits>

static inline void un_permute_custom_check(const at::Tensor& expert_output,
                                           const at::Tensor& topk_vals,
                                           const at::Tensor& inv_perm)
{
    TORCH_CHECK(expert_output.defined() && topk_vals.defined() && inv_perm.defined(),
                "un_permute_custom: inputs must be defined");

    TORCH_CHECK(expert_output.device().type() == c10::DeviceType::PrivateUse1,
                "un_permute_custom: expert_output must be on NPU");
    TORCH_CHECK(topk_vals.device().type() == c10::DeviceType::PrivateUse1,
                "un_permute_custom: topk_vals must be on NPU");
    TORCH_CHECK(inv_perm.device().type() == c10::DeviceType::PrivateUse1,
                "un_permute_custom: inv_perm must be on NPU");

    TORCH_CHECK(expert_output.scalar_type() == at::kBFloat16,
                "un_permute_custom: expert_output must be bfloat16");
    TORCH_CHECK(topk_vals.scalar_type() == at::kBFloat16,
                "un_permute_custom: topk_vals must be bfloat16");
    TORCH_CHECK(inv_perm.scalar_type() == at::kLong,
                "un_permute_custom: inv_perm must be int64");

    TORCH_CHECK(expert_output.is_contiguous(),
                "un_permute_custom: expert_output must be contiguous");
    TORCH_CHECK(topk_vals.is_contiguous(),
                "un_permute_custom: topk_vals must be contiguous");
    TORCH_CHECK(inv_perm.is_contiguous(),
                "un_permute_custom: inv_perm must be contiguous");

    TORCH_CHECK(expert_output.dim() == 2,
                "un_permute_custom: expert_output must be 2D [total_expanded, K]");
    TORCH_CHECK(topk_vals.dim() == 2,
                "un_permute_custom: topk_vals must be 2D [M, topk]");
    TORCH_CHECK(inv_perm.dim() == 1,
                "un_permute_custom: inv_perm must be 1D [M*topk]");

    const int64_t totalExp = expert_output.size(0);
    const int64_t K = expert_output.size(1);
    const int64_t M = topk_vals.size(0);
    const int64_t topk = topk_vals.size(1);

    TORCH_CHECK(M > 0 && topk > 0 && K > 0 && totalExp > 0,
                "un_permute_custom: empty dimensions not supported");

    TORCH_CHECK(inv_perm.numel() == M * topk,
                "un_permute_custom: inv_perm numel must equal M*topk, got ",
                inv_perm.numel(), " vs ", (M * topk));

    TORCH_CHECK(topk <= 8, "un_permute_custom: topk too large (max 8), got ", topk);
    TORCH_CHECK(K <= 4096, "un_permute_custom: hidden_size K too large (max 4096), got ", K);

    TORCH_CHECK(totalExp <= (int64_t)std::numeric_limits<int32_t>::max(),
                "un_permute_custom: total_expanded too large for this kernel");
}

at::Tensor un_permute_custom_impl_npu(const at::Tensor& expert_output,
                                      const at::Tensor& topk_vals,
                                      const at::Tensor& inv_perm)
{
    un_permute_custom_check(expert_output, topk_vals, inv_perm);

    const int64_t M = topk_vals.size(0);
    const int64_t K = expert_output.size(1);

    at::Tensor out = at::empty({M, K}, expert_output.options());
    EXEC_NPU_CMD(aclnnUnPermuteCustom, expert_output, topk_vals, inv_perm, out);
    return out;
}

TORCH_LIBRARY(myops, m) {
    m.def("un_permute_custom(Tensor expert_output, Tensor topk_vals, Tensor inv_perm) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("un_permute_custom", &un_permute_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("un_permute_custom", &un_permute_custom_impl_npu,
          "UnPermuteCustom (NPU, bf16): out[m,:] = sum_t topk_vals[m,t] * expert_output[inv_perm[m*topk+t], :]");
}
