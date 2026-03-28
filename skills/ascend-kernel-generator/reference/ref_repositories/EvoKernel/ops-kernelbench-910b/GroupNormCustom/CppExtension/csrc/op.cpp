
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor group_norm_custom_impl_npu(const at::Tensor& x,
                                      const at::Tensor& gamma,
                                      const at::Tensor& beta) {
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnGroupNormCustom, x, gamma, beta, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("group_norm_custom", &group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_norm_custom", &group_norm_custom_impl_npu,
          "group_norm_custom(x, gamma, beta) -> GroupNorm over dim=1 with baked eps=1e-5 and baked num_groups=8");
}
