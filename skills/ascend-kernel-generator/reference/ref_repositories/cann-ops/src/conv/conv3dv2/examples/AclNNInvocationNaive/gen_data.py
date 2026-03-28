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
import torch.nn.functional as F
import numpy as np

def gen_golden_data_simple():
    input_shape = [2, 2, 2, 2, 2]
    weight_shape = [1, 2, 1, 1, 1]

    input = torch.rand(input_shape, dtype=torch.float32) * 60 + 1 # range from 1 to 61
    weight = torch.rand(weight_shape, dtype=torch.float32) * 60 + 1 # range from 1 to 61

    strides = [1, 1, 1]
    pads = [1, 1, 1]
    dilations = [1, 1, 1]

    golden = F.conv3d(input, weight, stride=strides, padding=pads, dilation=dilations)

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input.numpy().tofile("./input/input.bin")
    weight.numpy().tofile("./input/weight.bin")
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

