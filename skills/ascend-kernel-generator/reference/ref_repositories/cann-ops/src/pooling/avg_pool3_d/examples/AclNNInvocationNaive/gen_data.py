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


def avgpool3d_golden(x, ksize, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None):
    x = torch.from_numpy(x)
    x = x.to(torch.float)
    model = torch.nn.AvgPool3d(
                                ksize, 
                                stride,
                                padding,
                                ceil_mode,
                                count_include_pad,
                                divisor_override
                                )
    output = model(x)
    return output.numpy()


def gen_golden_data_simple():
    dtype = np.float32
    input_shape = [1, 16, 4, 4, 4]
    ksize = [4, 4, 4]
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    output_shape = [1, 16, 1, 1, 1]

    x = np.random.uniform(-1, 1, input_shape).astype(dtype)

    golden = avgpool3d_golden(x, ksize, stride, padding, ceil_mode, count_include_pad, divisor_override)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input.bin")
    golden.astype(dtype).tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

