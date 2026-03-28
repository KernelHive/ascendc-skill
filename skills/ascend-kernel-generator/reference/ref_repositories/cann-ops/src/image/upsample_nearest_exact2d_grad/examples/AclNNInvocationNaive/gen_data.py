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


def upsample_nearest1d_backward():
    # 输入张量UpsampleNearest1dBackward
    input_shape = (1, 1, 3)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)
    input_tensor.requires_grad = True
    # 插值参数
    output_size = [2]    # 输出shape
    mode = 'nearest'    # 插值模式
    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode)
    output_grad = torch.tensor([1, 4.1]).reshape(1, 1, 2).type(torch.float32)
    output_tensor.backward(output_grad)
    input_grad = input_tensor.grad
    input_grad.numpy().tofile("./output/golden1.bin")


def upsample_nearest2d_backward():
    # 输入张量UpsampleNearest2dBackward
    input_shape = (2, 2, 1, 1)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)
    input_tensor.requires_grad = True
    # 插值参数
    output_size = [3, 3]    # 输出shape
    mode = 'nearest'    # 插值模式
    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode)
    output_grad = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]).reshape(2, 2, 3, 3).type(torch.float32)
    output_tensor.backward(output_grad)
    input_grad = input_tensor.grad
    input_grad.numpy().tofile("./output/golden2.bin")


def upsample_nearest_exact2d_backward():
    # 输入张量UpsampleNearestExact2dBackward
    input_shape = (1, 1, 8, 4)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)
    input_tensor.requires_grad = True
    # 插值参数
    output_size = [4, 2]    # 输出shape
    mode = 'nearest-exact'    # 插值模式
    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode)
    output_grad = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).reshape(1, 1, 4, 2).type(torch.float32)
    output_tensor.backward(output_grad)
    input_grad = input_tensor.grad
    input_grad.numpy().tofile("./output/golden3.bin")


def upsample_nearest_exact1d_backward():
    # 输入张量UpsampleNearestExact1dBackward
    input_shape = (1, 1, 4)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)
    input_tensor.requires_grad = True
    # 插值参数
    output_size = [2]    # 输出shape
    mode = 'nearest-exact'    # 插值模式
    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode)
    output_grad = torch.tensor([0, 1]).reshape(1, 1, 2).type(torch.float32)
    output_tensor.backward(output_grad)
    input_grad = input_tensor.grad
    input_grad.numpy().tofile("./output/golden4.bin")


def gen_golden_data_simple():
    upsample_nearest1d_backward()
    upsample_nearest2d_backward()
    upsample_nearest_exact2d_backward()
    upsample_nearest_exact1d_backward()

if __name__ == "__main__":
    gen_golden_data_simple()
