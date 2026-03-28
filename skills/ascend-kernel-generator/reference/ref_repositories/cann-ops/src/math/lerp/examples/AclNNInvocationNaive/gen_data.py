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
    start = np.random.uniform(-10, 10, [32]).astype(np.float16)
    end = np.random.uniform(-10, 10, [32]).astype(np.float16)
    weight = np.random.uniform(0, 1, [32]).astype(np.float16)
    res = torch.lerp(torch.Tensor(start), torch.Tensor(end), torch.Tensor(weight))
    
    golden = res.numpy().astype(np.float16)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    start.tofile("./input/input_start.bin")
    end.tofile("./input/input_end.bin")
    weight.tofile("./input/input_weight.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()