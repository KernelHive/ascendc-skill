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
    dtype = np.float16
    output_shape = [100, 100]
    input_data = np.random.uniform(1, 4, output_shape).astype(dtype)
    input_x1 = np.random.uniform(1, 4, output_shape).astype(dtype)
    input_x2 = np.random.uniform(1, 4, output_shape).astype(dtype)
    input_value = np.random.uniform(1, 4, [1]).astype(dtype)
    if dtype == np.int32:
        input_value[0] = 2
    else:
        input_value[0] = 6.2
    golden = (input_data + input_x1 / input_x2 * input_value).astype(dtype)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_data.tofile("./input/input_data.bin")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    input_value.tofile("./input/input_value.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
