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
    x1_shape = [5, 2]
    x2_shape = [2, 3]
    bias_shape = [3]
    pertokenScale_shape = [5]
    scale_shape = [3]
    output_shape = [5, 3]

    x1 = np.random.uniform(-1, 1, x1_shape).astype(np.int8)
    x2 = np.random.uniform(-1, 1, x2_shape).astype(np.int8)
    bias = np.random.uniform(-1, 1, bias_shape).astype(np.int32)
    pertokenScale = np.random.uniform(0.0, 1.0, pertokenScale_shape).astype(np.float32) / 1000
    scale = np.random.uniform(0.0, 1.0, scale_shape).astype(np.float32) / 1000

    golden = ((x1.astype(np.float32) @ x2.astype(np.float32)).astype(np.int32) + bias).astype(np.float32)
    golden = golden * scale
    golden = golden * pertokenScale.reshape(-1, 1)
    golden = golden.astype(np.float16)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1.astype(np.int8).tofile("./input/input_x1.bin")
    x2.astype(np.int8).tofile("./input/input_x2.bin")
    bias.astype(np.int32).tofile("./input/input_bias.bin")
    pertokenScale.astype(np.float32).tofile("./input/input_pertokenScale.bin")
    scale.astype(np.float32).tofile("./input/input_scale.bin")

    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


