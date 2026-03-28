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

def gen_golden_data_simple():
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    dtype = np.float16
    var_ref_shape = [4, 3]
    input_value_shape = [2, 2]

    var_ref = np.random.uniform(-1, 1, var_ref_shape).astype(dtype)
    input_value = np.random.uniform(-1, 1, input_value_shape).astype(dtype)

    var_ref.astype(dtype).tofile("./input/input0.bin")
    input_value.astype(dtype).tofile("./input/input1.bin")
    # 切片参数
    begin = [1, 0]
    end = [4, 2]
    strides = [2, 1]

    # 构造切片对象
    row_slice = slice(begin[0], end[0], strides[0])  # 1:4:2 → 行索引 [1, 3]
    col_slice = slice(begin[1], end[1], strides[1])  # 0:2:1 → 列索引 [0, 1]

    # 执行赋值
    var_ref[row_slice, col_slice] = input_value
    
    golden = var_ref

    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
