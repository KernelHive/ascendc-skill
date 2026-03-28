
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor min_gpt_new_gelu_custom_impl_npu(const at::Tensor& x) {
    at::Tensor y = at::empty_like(x);
    EXEC_NPU_CMD(aclnnMinGptNewGeluCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("min_gpt_new_gelu_custom", &min_gpt_new_gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_gpt_new_gelu_custom", &min_gpt_new_gelu_custom_impl_npu,
          "min_gpt_new_gelu_custom(x) -> GPT/BERT GELU (tanh approximation), fused AscendC op");
}
