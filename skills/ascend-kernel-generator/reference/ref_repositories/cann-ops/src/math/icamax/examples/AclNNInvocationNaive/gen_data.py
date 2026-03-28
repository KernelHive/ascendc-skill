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
    dtype = np.complex64
    input_shape = [2046]

    x_r =  np.random.uniform(-1, 1, input_shape).astype(np.float32)
    x_i =  np.random.uniform(-1, 1, input_shape).astype(np.float32)
    x = (x_r + 1j * x_i).astype(dtype)
    y = np.ones([1]).astype(np.int32)

    golden = np.argmax(np.abs(np.real(x)) + np.abs(np.imag(x))) + 1
    print(f"input x: {x}")
    print(f"golden: {golden}")

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_x.bin")
    y.astype(np.int32).tofile("./input/input_y.bin")
    golden.astype(np.int32).tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()