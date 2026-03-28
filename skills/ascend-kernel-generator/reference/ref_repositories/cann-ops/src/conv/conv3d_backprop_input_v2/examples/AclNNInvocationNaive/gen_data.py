#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import torch
import numpy as np

def gen_golden_data_simple():
    gradOutputShape = [2, 2, 7, 7, 7]
    inputShape = [2, 2, 7, 7, 7]
    weightShape = [2, 2, 1, 1, 1]
    biasSize = [2]
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    dilation = [1, 1, 1]
    transposed = False
    outputPadding = [0, 0, 0]
    groups = 1
    outputMask = [True, False, False]

    out_backprop = torch.rand(gradOutputShape, dtype=torch.float32)
    input = torch.rand(inputShape, dtype=torch.float32)
    filter = torch.rand(weightShape, dtype=torch.float32)

    grad_input, _, _ = torch.ops.aten.convolution_backward(
        out_backprop,
        input,
        filter,
        bias_sizes=biasSize,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=outputPadding,
        groups=groups,
        output_mask=outputMask
    )

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    out_backprop.numpy().tofile("./input/out_backprop.bin")
    input.numpy().tofile("./input/input.bin")
    filter.numpy().tofile("./input/filter.bin")
    grad_input.numpy().tofile("./output/grad_input_cpu.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

