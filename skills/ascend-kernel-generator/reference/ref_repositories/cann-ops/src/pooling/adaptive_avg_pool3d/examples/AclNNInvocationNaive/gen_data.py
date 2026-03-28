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
    # 输入张量
    input_self = torch.rand(2, 1, 2, 2, 4) * 20 - 10    # value range: (-10, 10), shape: (2, 1, 2, 2, 4)
    input_self = input_self.to(torch.float32)  # dtype: float32
    input_output_size = [2, 2, 2]

    # 调用函数
    golden_out = torch._adaptive_avg_pool3d(
        input_self, input_output_size
    )

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_self.numpy().tofile("./input/input_self.bin")
    golden_out.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    torch.manual_seed(2025)
    gen_golden_data_simple()

