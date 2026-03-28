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
import math
import numpy as np


def log(x, base: float = -1.0, scale: float = 1.0, shift: float = 0.0):
    # 转换为 PyTorch 张量
    exponent = x * scale + shift
    epsilon = 1e-15
    if abs(base - (-1.0)) > epsilon:
        return np.log(exponent) / math.log(base)
    return np.log(exponent)

def gen_golden_data_simple():
    dtype = np.float16
    output_shape = [100, 100]
    input_x = np.random.uniform(1, 4, output_shape).astype(dtype)
    
    # 计算 golden 数据
    golden = log(input_x, -1.0, 5.0, 5.0).astype(dtype)

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()