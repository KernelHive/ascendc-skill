
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

static inline void emsa_check_caps(int64_t NQ, int64_t NK, int64_t DK, int64_t DV) {
    TORCH_CHECK(NQ <= 64, "emsa_custom: NQ>64 not supported by this kernel");
    TORCH_CHECK(NK <= 64, "emsa_custom: NK>64 not supported by this kernel");
    TORCH_CHECK(DK <= 64, "emsa_custom: DK>64 not supported by this kernel");
    TORCH_CHECK(DV <= 64, "emsa_custom: DV>64 not supported by this kernel");
}

at::Tensor emsa_custom_impl_npu(const at::Tensor& q,
                               const at::Tensor& k,
                               const at::Tensor& v) {
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "emsa_custom: q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "emsa_custom: k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "emsa_custom: v must be on NPU");

    TORCH_CHECK(q.scalar_type() == at::kFloat, "emsa_custom: only float32 supported");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "emsa_custom: only float32 supported");
    TORCH_CHECK(v.scalar_type() == at::kFloat, "emsa_custom: only float32 supported");

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "emsa_custom: expected q/k/v to be 4D");

    TORCH_CHECK(q.is_contiguous(), "emsa_custom: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "emsa_custom: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "emsa_custom: v must be contiguous");

    const auto B  = q.size(0);
    const auto H  = q.size(1);
    const auto NQ = q.size(2);
    const auto DK = q.size(3);

    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(2) == DK,
                "emsa_custom: k must be [B,H,DK,NK] and match q's B,H,DK");
    const auto NK = k.size(3);

    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == NK,
                "emsa_custom: v must be [B,H,NK,DV] and match k's B,H,NK");
    const auto DV = v.size(3);

    TORCH_CHECK(B > 0 && H > 0 && NQ > 0 && DK > 0 && NK > 0 && DV > 0,
                "emsa_custom: all dimensions must be > 0");

    emsa_check_caps(NQ, NK, DK, DV);

    at::Tensor y = at::empty({B, H, NQ, DV}, q.options());
    EXEC_NPU_CMD(aclnnEMSACustom, q, k, v, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("emsa_custom", &emsa_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("emsa_custom", &emsa_custom_impl_npu,
          "EMSACustom fused attention core: y=softmax((q@k)/sqrt(DK))@v (NPU, float32)");
}
