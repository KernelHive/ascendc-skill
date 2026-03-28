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
import torch.nn.functional as F

def gen_golden_data_simple():
    dy = np.random.uniform(-10, 10, [1024]).astype(np.float16)
    input_x = np.random.uniform(-10, 10, [1024]).astype(np.float16)
    

    attr = 1.702
    attr_opp = 0 - attr
    attr_half = attr / 2

    abs_x = np.abs(input_x)
    mul_abs_x = abs_x * attr_opp
    exp_x = np.exp(mul_abs_x)

    add_2 = input_x * exp_x * attr
    exp_pn_x = np.exp((input_x - abs_x) * attr)

    div_up = exp_x + add_2 +exp_pn_x
    div_down = (exp_x + 1) ** 2

    res = div_up / div_down
    golden = dy * res

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    dy.tofile("./input/input_dy.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()