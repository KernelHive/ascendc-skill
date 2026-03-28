#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

from typing import Any
import torch
from torch.library import impl
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
from torch_npu.meta._meta_registrations import m


@impl(m, "npu_add_custom")
def npu_add_custom_meta(x, y):
    return torch.empty_like(x)


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.npu.npu_add_custom.default)
def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "AddCustom",
        inputs={
            "x": x,
            "y": y,
        },
        outputs=['z']
    )


class TestTorchCompileCustomAdd(TestCase):

    def test_add_custom(self):
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        length = [8, 2048]
        x = torch.rand(length, device='npu', dtype=torch.float16)
        y = torch.rand(length, device='npu', dtype=torch.float16)
        print(x, '\n', y)
        
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch_npu.npu_add_custom(x, y)
        mod = torch.compile(Module().npu(), backend=npu_backend)
        output = mod(x, y)
        print(output)
        self.assertRtolEqual(output, (x + y))


if __name__ == "__main__":
    run_tests()
