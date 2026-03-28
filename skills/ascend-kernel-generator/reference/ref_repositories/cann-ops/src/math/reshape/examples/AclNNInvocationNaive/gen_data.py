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
import numpy as np
import torch


def reshape_tensor(x, shape_tensor, axis=0, num_axes=-1):
    original_dims = len(x.shape)
    if num_axes == -1:
        num_axes = original_dims - axis
    left_shape = list(x.shape[:axis])
    right_shape = list(x.shape[axis + num_axes:])
    new_middle_shape = shape_tensor.tolist()
    new_shape = left_shape + new_middle_shape + right_shape
    reshaped_x = x.reshape(new_shape)
    return reshaped_x


def gen_golden_data_simple():
    length_x = [8, 47, 123, 1023]
    x = (torch.rand(length_x, device='cpu') * 10 - 5).to(torch.float32)
    shape = torch.tensor([123, 47, 8, 1023], dtype=torch.int32)
    axis = 0
    num_axes = -1
    golden = reshape_tensor(x, shape, axis, num_axes).numpy()

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.numpy().tofile("./input/input_x.bin")
    shape.numpy().tofile("./input/input_shape.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

