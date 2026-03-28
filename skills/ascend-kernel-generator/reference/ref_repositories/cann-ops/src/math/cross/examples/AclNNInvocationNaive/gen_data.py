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
    x1_tensor = np.random.uniform(-10, 10, [4, 3]).astype(np.int8)
    x2_tensor = np.random.uniform(-10, 10, [4, 3]).astype(np.int8)
    x1 = torch.from_numpy(x1_tensor)
    x2 = torch.from_numpy(x2_tensor)
    dim = 1
    res_tensor = torch.cross(x1, x2)
    golden = res_tensor.numpy().astype(np.int8)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_tensor.tofile("./input/input_x1.bin")
    x2_tensor.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()