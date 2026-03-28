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
from torch.nn.functional import interpolate


def gen_golden_data_simple():
    # 输入张量UpsampleNearest3d
    input_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                     43, 44, 45, 46, 47, 48]).reshape(2, 2, 2, 2, 3).type(torch.float32)
    # 插值参数
    output_size = [1, 1, 1]    # 输出shape
    mode = 'nearest'    # 插值模式

    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode)
    output_tensor.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

