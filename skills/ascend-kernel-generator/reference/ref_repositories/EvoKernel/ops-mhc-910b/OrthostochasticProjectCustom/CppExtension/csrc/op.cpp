
#include <torch/library.h>
#include <torch/extension.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor orthostochastic_project_custom_impl_npu(
    const at::Tensor& logits,
    int64_t steps,
    double eps,
    double a,
    double b,
    double c)
{
    TORCH_CHECK(logits.device().is_privateuseone(), "logits must be on NPU");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D (n,n)");
    TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0, "invalid shape");
    TORCH_CHECK(logits.size(0) <= 32 && logits.size(1) <= 32, "only supports up to 32x32");

    at::Tensor x = logits.contiguous();
    if (x.scalar_type() != at::kFloat) x = x.to(at::kFloat);

    // Keep scalar params as device tensors (1-element) to match op signature.
    at::Tensor steps_t = at::empty({1}, x.options().dtype(at::kInt));
    steps_t.fill_(static_cast<int32_t>(steps));

    at::Tensor eps_t = at::empty({1}, x.options().dtype(at::kFloat));
    eps_t.fill_(static_cast<float>(eps));

    auto make_scalar_f32 = [&](double v) -> at::Tensor {
        at::Tensor t = at::empty({1}, x.options().dtype(at::kFloat));
        t.fill_(static_cast<float>(v));
        return t;
    };

    at::Tensor a_t = make_scalar_f32(a);
    at::Tensor b_t = make_scalar_f32(b);
    at::Tensor c_t = make_scalar_f32(c);

    at::Tensor out = at::empty_like(x);

    EXEC_NPU_CMD(aclnnOrthostochasticProjectCustom,
                 x, steps_t, eps_t, a_t, b_t, c_t,
                 out);

    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("orthostochastic_project_custom", &orthostochastic_project_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("orthostochastic_project_custom", &orthostochastic_project_custom_impl_npu,
          "orthostochastic_project_custom: Newton–Schulz orthostochastic projection + square (NPU AscendC)");
}
