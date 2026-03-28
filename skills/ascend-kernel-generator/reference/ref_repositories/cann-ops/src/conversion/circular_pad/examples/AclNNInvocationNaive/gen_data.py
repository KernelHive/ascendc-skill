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
    dtype = torch.float
    shape = (1, 1, 2, 2)
    values = [1, 2, 3, 4]

    input_tensor = torch.tensor(values, dtype=torch.float).reshape(shape)

#调用函数
    golden = torch.nn.functional.pad(input_tensor, pad=(1, 1, 1, 1), mode='circular')  # 对最后两个维度进行padding
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_tensor.numpy().tofile("./input/input.bin")
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

