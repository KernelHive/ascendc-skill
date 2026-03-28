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

import numpy as np
import os


def gen_golden_data():
    m_shape = 32
    n_shape = 32
    k_shape = 32

    x1_gm = np.random.randint(1, 10, [m_shape, k_shape]).astype(np.float16)
    x2_gm = np.random.randint(1, 10, [k_shape, n_shape]).astype(np.float16)
    bias_gm = np.random.randint(1, 10, [n_shape]).astype(np.float32)
    temp_y = np.matmul(x1_gm.astype(np.float16), x2_gm.astype(np.float16))
    golden = temp_y.astype(np.float32) + bias_gm
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
