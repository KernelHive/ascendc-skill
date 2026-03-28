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
    a1 = np.random.uniform(-0.2, 0.2, [40, 64, 32]).astype(dtype)
    a2 = np.random.uniform(-0.2, 0.2, [1, 1, 32]).astype(dtype)
    a3 = np.array([-0.1]).astype(dtype)
    a4 = np.random.uniform(-0.2, 0.2, [40, 64, 1]).astype(dtype)
    a5 = np.random.uniform(-0.2, 0.2, [40, 64, 32]).astype(dtype)

    add_res = a1 + a2
    mul1_res = add_res * a3
    sig_res = 1 / (1 + np.exp(-mul1_res))
    mul2_res = sig_res * a4
    mul3_res = mul2_res * a5
    np_res = np.sum(mul3_res, axis=1)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    a1.astype(dtype).tofile("./input/input_1.bin")
    a2.astype(dtype).tofile("./input/input_2.bin")
    a3.astype(dtype).tofile("./input/input_3.bin")
    a4.astype(dtype).tofile("./input/input_4.bin")
    a5.astype(dtype).tofile("./input/input_5.bin")
    np_res.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

