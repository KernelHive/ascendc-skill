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
    tf.compat.v1.disable_eager_execution()

    mul0input0 = np.random.rand(90, 1024).astype(np.float16)
    mul0input1 = np.random.rand(90, 1024).astype(np.float16)
    mul1input0 = np.float16(1.0)
    addy = np.float16(1.0)
    gamma = np.random.rand(1, 1024).astype(np.float16)
    beta = np.random.rand(1, 1024).astype(np.float16)

    mul_res = mul0input0 * mul0input1 * mul1input0
    reduce_mean_0 = np.mean(mul_res, axis=1, keepdims=True)
    diff = mul_res - reduce_mean_0
    muld_res = diff * diff
    x2 = np.mean(muld_res, axis=1, keepdims=True)
    reduce_mean_1 = gamma / np.sqrt(x2 + addy)
    output = beta - reduce_mean_1 * reduce_mean_0 + reduce_mean_1 * mul_res

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    mul0input0.tofile("./input/mul0input0.bin")
    mul0input1.tofile("./input/mul0input1.bin")
    np.array([mul1input0], dtype=np.float16).tofile("./input/mul1input0.bin")
    np.array([addy], dtype=np.float16).tofile("./input/addy.bin")
    gamma.tofile("./input/gamma.bin")
    beta.tofile("./input/beta.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()