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
    a1 = np.random.randn(32, 32).astype(dtype)
    a2 = np.array([1.2]).astype(dtype)
    a3 = np.array([1.3]).astype(dtype)
    a4 = np.array([1.4]).astype(dtype)

    mul1_res = a1 * a2
    sigmoid_res = 1 / (1 + np.exp(-mul1_res))
    mul_2_res = sigmoid_res * a3
    result = mul_2_res + a4

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    a1.astype(dtype).tofile("./input/input_1.bin")
    a2.astype(dtype).tofile("./input/input_2.bin")
    a3.astype(dtype).tofile("./input/input_3.bin")
    a4.astype(dtype).tofile("./input/input_4.bin")
    result.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

