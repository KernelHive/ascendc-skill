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

    a1 = np.random.randn(25, 32 * 1024).astype(dtype)
    a2 = np.random.randn(1, 256, 128).astype(dtype)
    t1 = float(0.3)
    t2 = float(0.1)
    t3 = float(0.8)
    tmp = 1 / (1 + np.exp(-a1 * t1))
    zero = np.zeros_like(tmp)
    sel = np.where(tmp < t2, tmp, 2 * tmp)
    sel = sel.reshape(-1, 32 * 1024) * a2.reshape(1, 32 * 1024)
    res = sel * t3
    numpy_result = res.reshape(res.shape[0], 256, 128).astype(dtype)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    a1.astype(dtype).tofile("./input/input_1.bin")
    a2.astype(dtype).tofile("./input/input_2.bin")
    numpy_result.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

