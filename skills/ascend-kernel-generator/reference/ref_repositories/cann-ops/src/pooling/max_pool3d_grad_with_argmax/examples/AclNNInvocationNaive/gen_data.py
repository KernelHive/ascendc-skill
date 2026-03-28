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
    dtype = np.float32
    gradout_shape = [1, 1, 1, 1, 1]
    self_shape = [1, 1, 2, 2, 2]
    indices_shape = [1, 1, 1, 1, 1]
    out_shape = [1, 1, 2, 2, 2]

    kernel_data = [2, 2, 2]
    stride_data = [2, 2, 2]
    padding_data = [0, 0, 0]
    dilation_data = [1, 1, 1]

    input_gradout = np.random.uniform(1, 10, gradout_shape).astype(dtype)
    input_self = np.random.uniform(1, 10, self_shape).astype(dtype)
    input_indices = np.random.uniform(1, 10, indices_shape).astype(np.int32)
    # 将 NumPy 数组转换为 PyTorch 张量
    input_gradout_tensor = torch.tensor(input_gradout, dtype=torch.float32)
    input_self_tensor = torch.tensor(input_self, dtype=torch.float32)
    input_indices_tensor = torch.tensor(input_indices, dtype=torch.int64)

    output = torch.ops.aten.max_pool3d_with_indices_backward(
        input_gradout_tensor, 
        input_self_tensor,
        kernel_data, 
        stride_data, 
        padding_data, 
        dilation_data,
        False,
        input_indices_tensor
        )

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_gradout.tofile("./input/gradout.bin")
    input_self.tofile("./input/self.bin")
    input_indices.tofile("./input/indices.bin")

    output.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

