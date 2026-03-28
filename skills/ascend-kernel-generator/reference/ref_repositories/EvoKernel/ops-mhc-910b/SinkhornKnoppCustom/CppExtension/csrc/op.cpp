
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor sinkhorn_knopp_impl_npu(const at::Tensor& logits, int64_t tmax, double eps, double clamp_min)
{
    at::Tensor x = logits;
    if (x.scalar_type() != at::kFloat) {
        x = x.to(at::kFloat);
    }
    at::Tensor out = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSinkhornKnoppCustom, x, tmax, eps, clamp_min, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sinkhorn_knopp_custom", &sinkhorn_knopp_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_knopp_custom", &sinkhorn_knopp_impl_npu, "Sinkhorn-Knopp normalization (AscendC)");
}
