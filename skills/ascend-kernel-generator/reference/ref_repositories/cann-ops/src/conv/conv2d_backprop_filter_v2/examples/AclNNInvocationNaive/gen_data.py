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
    input_shape = [1024, 3, 224, 224]
    weight_shape = [1024, 3, 16, 16]

    input = torch.ones(input_shape, dtype=torch.float32) 
    weight = torch.ones(weight_shape, dtype=torch.float32) 

    weight.requires_grad_(True)
    strides = [16, 16]
    pads = [0, 0]
    dilations = [1, 1]

    bias = torch.ones([1024], dtype=torch.float32)
    golden = F.conv2d(input, weight, bias=bias, stride=strides, padding=pads, dilation=dilations)

    grad_output = torch.ones_like(golden)
    golden.backward(grad_output)
    output = weight.grad

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_f32 = input.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    grad_output_f32 = grad_output.to(torch.float32)
    output_f32 = output.to(torch.float32)

    input_fp16 = input_f32.detach().numpy().astype(np.float16)
    weight_fp16 = weight_f32.detach().numpy().astype(np.float16)
    grad_output_fp16 = grad_output_f32.detach().numpy().astype(np.float16)

    input_fp16.tofile("./input/input.bin")
    weight_fp16.tofile("./input/weight.bin")
    grad_output_fp16.tofile("./input/gradOutput.bin")
    output_f32.detach().numpy().tofile("./output/output.bin")


if __name__ == "__main__":
    gen_golden_data_simple()