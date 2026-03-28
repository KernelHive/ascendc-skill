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
import numpy as np
import torch

from torch.nn.functional import interpolate


def gen_golden_data_simple():
    # 输入张量
    input_shape = (1, 1, 2, 2, 2)
    # value range: (-255, 255), shape: (1, 1, 2, 5) dtype: float32
    input_tensor = torch.randn(input_shape, dtype=torch.float32) * 255

    # 插值参数
    output_size = [4, 4, 4]    # 输出shape
    mode = 'trilinear'    # 插值模式
    align_corners = False    # 角对齐

    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode, align_corners=align_corners)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_tensor.numpy().tofile("./input/input_tensor.bin")
    output_tensor.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
