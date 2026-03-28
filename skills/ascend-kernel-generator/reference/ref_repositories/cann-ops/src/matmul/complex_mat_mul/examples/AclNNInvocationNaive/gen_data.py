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
    dtype = np.complex64
    batch_size = 4
    m, k, n = 300, 400, 500

    # Generate random complex matrices
    input_x = (np.random.randn(batch_size, m, k) + 1j * np.random.randn(batch_size, m, k)).astype(dtype)
    input_y = (np.random.randn(batch_size, k, n) + 1j * np.random.randn(batch_size, k, n)).astype(dtype)
    bias = (np.random.randn(batch_size, m, n) + 1j * np.random.randn(batch_size, m, n)).astype(dtype)

    golden = np.matmul(input_x, input_y) + bias

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    bias.tofile("./input/input_bias.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
