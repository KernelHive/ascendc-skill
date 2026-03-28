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
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def gen_golden_data_simple():
    input_shape = [4, 320, 80, 80]
    weight_shape = [320, 320, 3, 3]
    bias_shape = [320]

    input = torch.ones(input_shape, dtype=torch.float32)
    weight = torch.ones(weight_shape, dtype=torch.float32)
    bias = torch.ones(bias_shape, dtype=torch.float32)

    input.requires_grad_(True)
    strides = [1, 1]
    pads = [1, 1]
    dilations = [1, 1]

    y = F.conv2d(input, weight, bias=bias, stride=strides, padding=pads, dilation=dilations)

    gradOutput = torch.ones_like(y)
    y.backward(gradOutput)
    output = input.grad
    
    output_f32 = output.to(torch.float32)
    output_f32.detach().numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
