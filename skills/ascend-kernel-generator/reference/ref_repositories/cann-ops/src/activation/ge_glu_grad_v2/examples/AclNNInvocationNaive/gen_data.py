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

def gen_golden_data_simple():
    tensor_x = torch.randn(2, 2).to(torch.float32)
    tensor_dy = torch.randn(2, 1).to(torch.float32)
    tensor_x.requires_grad = True
    
    x, gate = tensor_x.chunk(2, dim=-1)
    x, gate = gate, x
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    y_gelu = torch.nn.functional.gelu(gate, approximate='tanh')
    
    tensor_x.clone().detach().numpy().tofile(f"./input/input_x.bin")
    tensor_dy.clone().detach().numpy().tofile(f"./input/input_dy.bin")
    y_gelu.clone().detach().numpy().tofile(f"./input/input_gelu.bin")

    y = x * y_gelu
    y.backward(tensor_dy)
    x_grad = tensor_x.grad.numpy()
    
    x_grad.tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

