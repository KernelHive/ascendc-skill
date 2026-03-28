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

import torch
import torch.nn.functional as F
import numpy as np
import os

def swish(x):
    return x * torch.sigmoid(x)

def swish_grad(x):
    return torch.sigmoid(x) + x * (1 - torch.sigmoid(x)) * torch.sigmoid(x)

def gen_golden_data_simple():
    self = torch.randn(2, 32).to(torch.float32)
    
    dim = -1

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    self.numpy().astype(np.float32).tofile("./input/input_x.bin")
    x = torch.chunk(self, 2, dim=dim)
    x0 = x[0].type(torch.float32)
    x1 = x[1].type(torch.float32)

    grad_output = torch.randn(2, 16).to(torch.float32)  
    grad_output.numpy().astype(np.float32).tofile("./input/grad_output.bin")
    
    # 使用 swish 和 swish_grad 来替代 silu 和 silu_derivative
    sigmoid_x0 = torch.sigmoid(x0)
    swish_derivative = swish_grad(x0)  # 计算 swish 的导数
    
    # 反向传播计算
    grad_swish = grad_output * x1 * swish_derivative
    grad_x1 = grad_output * swish(x0)
    grad_x0 = grad_swish # 使用 swish_grad 来计算梯度

    # 合并梯度
    grad_input = torch.cat((grad_x0, grad_x1), dim=dim)  # 合并梯度
    
    # 保存反向传播的梯度
    grad_input.numpy().astype(np.float32).tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


