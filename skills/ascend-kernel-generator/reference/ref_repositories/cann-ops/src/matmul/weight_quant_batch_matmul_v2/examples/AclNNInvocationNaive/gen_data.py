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
    x_shape = [16, 32]
    weight_shape = [32, 16]
    antiquant_scale_shape = [16]
    output_shape = [16, 16]

    x = np.random.uniform(-1, 1, x_shape).astype(np.float16)
    weight = np.random.uniform(-1, 1, weight_shape).astype(np.int8)
    antiquant_scale = np.random.uniform(0.0, 1.0, antiquant_scale_shape).astype(np.float16) / 1000

    golden = x.astype(np.float16) @ (weight * antiquant_scale.astype(np.float16)).astype(np.float16)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(np.float16).tofile("./input/input_x.bin")
    weight.astype(np.int8).tofile("./input/input_weight.bin")
    antiquant_scale.astype(np.float16).tofile("./input/input_antiquantScale.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


