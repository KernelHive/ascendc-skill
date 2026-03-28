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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def do_gelu(x):
    alpha = torch.sqrt(torch.tensor(2.0 / torch.pi))
    beta = 0.044715
    temp_x = alpha * (x + beta * x * x * x)
    gelu_y = 0.5 * x * (1.0 + torch.tanh(temp_x))

    return gelu_y

def gen_golden_data_simple():
    input_data = torch.randn(2, 2).to(torch.float32)
    x, gate = input_data.chunk(2, dim=-1)
    gelu = do_gelu(gate)
    result = x * gelu

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_data.numpy().astype(np.float32).tofile("./input/input_x.bin")
    result.detach().numpy().tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

