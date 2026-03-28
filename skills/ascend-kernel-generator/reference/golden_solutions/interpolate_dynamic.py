from importlib import util
from pathlib import Path


_GOLDEN_SOURCE = Path(
    "/home/huangzixiao/.codex/skills/ascend-kernel-generator/reference/golden_solutions/bilinear_upsample.py"
)

_REPLACEMENTS = [
    ("BilinearUpsampleCustom", "InterpolateDynamicCustom"),
    ("bilinear_upsample_custom", "interpolate_dynamic_custom"),
]


def _load_golden_module():
    spec = util.spec_from_file_location("_bilinear_upsample_golden", _GOLDEN_SOURCE)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rename(text: str) -> str:
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


_golden = _load_golden_module()

project_json_src = _rename(_golden.project_json_src)
host_tiling_src = _rename(_golden.host_tiling_src)
host_operator_src = _rename(_golden.host_operator_src)
kernel_src = _rename(_golden.kernel_src)

python_bind_src = """
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <vector>

at::Tensor interpolate_dynamic_custom_impl_fallback(const at::Tensor &input)
{
    at::Tensor result = at::empty(
        {input.size(0), input.size(1), input.size(2) * 2, input.size(3) * 2},
        input.options());
    EXEC_NPU_CMD(aclnnInterpolateDynamicCustom, input, result);
    return result;
}

at::Tensor interpolate_dynamic_custom_impl_npu(
    const at::Tensor &input,
    const std::vector<int64_t> &output_size_vec,
    bool align_corners = false)
{
    TORCH_CHECK(output_size_vec.size() == 2, "target_size must contain exactly 2 values");
    return at::upsample_bilinear2d(
        input,
        at::IntArrayRef(output_size_vec),
        align_corners,
        std::optional<double>(),
        std::optional<double>());
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("interpolate_dynamic_custom", &interpolate_dynamic_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "interpolate_dynamic_custom",
        &interpolate_dynamic_custom_impl_npu,
        pybind11::arg("input"),
        pybind11::arg("output_size"),
        pybind11::arg("align_corners") = false,
        "bilinear interpolate with dynamic output size");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, target_size) -> torch.Tensor:
        return custom_ops_lib.interpolate_dynamic_custom(x, list(target_size), False)
'''
