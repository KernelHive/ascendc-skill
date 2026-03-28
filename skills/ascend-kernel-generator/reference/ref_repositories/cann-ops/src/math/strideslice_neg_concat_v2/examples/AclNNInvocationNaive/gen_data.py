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

    input0 = np.random.randn(1, 128, 1, 128).astype(dtype)

    batch_size, height, width, channels = input0.shape
    mid_col = channels // 2
    result = input0.copy()
    result[:, :, :, mid_col:] = -input0[:, :, :, mid_col:]

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input0.astype(dtype).tofile("./input/input_1.bin")
    result.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

