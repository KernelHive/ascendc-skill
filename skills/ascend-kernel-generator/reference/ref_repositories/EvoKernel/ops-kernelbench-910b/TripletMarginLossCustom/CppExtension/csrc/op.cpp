
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor triplet_margin_loss_impl_npu(const at::Tensor& anchor,
                                       const at::Tensor& positive,
                                       const at::Tensor& negative,
                                       const at::Tensor& margin)
{
    at::Tensor result = at::empty({}, anchor.options());
    EXEC_NPU_CMD(aclnnTripletMarginLossCustom, anchor, positive, negative, margin, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("triplet_margin_loss_custom", &triplet_margin_loss_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triplet_margin_loss_custom", &triplet_margin_loss_impl_npu, "TripletMarginLossCustom (NPU)");
}
