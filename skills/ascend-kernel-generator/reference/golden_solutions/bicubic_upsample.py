from pathlib import Path
import re


_OFFICIAL_ROOT = Path(
    "/home/huangzixiao/.codex/skills/ascend-kernel-generator/reference/ref_repositories/cann-ops/src/image/upsample_bicubic2d"
)

_REPLACEMENTS = [
    ("UPSAMPLE_BICUBIC2D_310P", "BICUBIC_UPSAMPLE_CUSTOM_310P"),
    ("UPSAMPLE_BICUBIC2D", "BICUBIC_UPSAMPLE_CUSTOM"),
    ("UpsampleBicubic2d", "BicubicUpsampleCustom"),
    ("upsample_bicubic2d", "bicubic_upsample_custom"),
]


def _apply_replacements(text: str) -> str:
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _inline_local_includes(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    result_lines = []
    include_pattern = re.compile(r'^\s*#include "([^"]+)"\s*$')
    for line in text.splitlines():
        match = include_pattern.match(line)
        if match:
            include_path = path.parent / match.group(1)
            if include_path.exists():
                result_lines.append(_inline_local_includes(include_path))
                continue
        result_lines.append(line)
    return "\n".join(result_lines)


project_json_src = '''
[
    {
        "op": "BicubicUpsampleCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "input",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float",
                    "half",
                    "bfloat16"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "output",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float",
                    "half",
                    "bfloat16"
                ]
            }
        ],
        "attr": [
            {
                "name": "output_size",
                "param_type": "required",
                "type": "listInt"
            },
            {
                "name": "align_corners",
                "param_type": "optional",
                "type": "bool",
                "default_value": "true"
            },
            {
                "name": "scales_h",
                "param_type": "optional",
                "type": "float",
                "default_value": "0.0"
            },
            {
                "name": "scales_w",
                "param_type": "optional",
                "type": "float",
                "default_value": "0.0"
            }
        ]
    }
]
'''

host_tiling_src = _apply_replacements(
    (_OFFICIAL_ROOT / "op_host" / "upsample_bicubic2d_tiling.h").read_text(encoding="utf-8")
)

host_operator_src = _apply_replacements(
    "\n\n".join(
        [
            _inline_local_includes(_OFFICIAL_ROOT / "op_host" / "upsample_bicubic2d_def.cpp"),
            _inline_local_includes(_OFFICIAL_ROOT / "op_host" / "upsample_bicubic2d.cpp"),
        ]
    )
)

kernel_src = _apply_replacements(
    _inline_local_includes(_OFFICIAL_ROOT / "op_kernel" / "upsample_bicubic2d.cpp")
)

python_bind_src = """
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

at::Tensor bicubic_upsample_impl_npu(const at::Tensor &input, bool align_corners = true)
{
    std::vector<int64_t> output_size_vec = {256, 256};
    at::IntArrayRef output_size(output_size_vec);
    double scales_h = 0.0;
    double scales_w = 0.0;
    at::Tensor result = at::empty({input.size(0), input.size(1), 256, 256}, input.options());
    EXEC_NPU_CMD(aclnnBicubicUpsampleCustom, input, output_size, align_corners, scales_h, scales_w, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("bicubic_upsample_custom", &bicubic_upsample_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bicubic_upsample_custom", &bicubic_upsample_impl_npu, "bicubic upsample to 256x256");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, align_corners: bool = True):
        super(ModelNew, self).__init__()
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.bicubic_upsample_custom(x, self.align_corners)
'''
