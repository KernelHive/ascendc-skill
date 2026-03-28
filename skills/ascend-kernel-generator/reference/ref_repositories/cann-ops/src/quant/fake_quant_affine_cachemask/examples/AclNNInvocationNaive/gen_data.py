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
    input_self = torch.rand(2, 5) * 20 - 10    # value range: (-10, 10), shape: (2, 5)
    input_self = input_self.to(torch.float32)  # dtype: float32

    # 量化参数（标量张量）
    scale = torch.tensor(0.1)       # 步长 0.1
    zero_point = torch.tensor(25)   # 零点 25
    fake_quant_enabled = torch.tensor(True)  # 启用伪量化

    # 调用函数
    output, mask = torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        input_self, scale, zero_point, fake_quant_enabled, quant_min=0, quant_max=50
    )

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_self.numpy().tofile("./input/input_self.bin")
    output.numpy().tofile("./output/golden_out.bin")
    mask.numpy().tofile("./output/golden_mask.bin")

if __name__ == "__main__":
    torch.manual_seed(2025)
    gen_golden_data_simple()

