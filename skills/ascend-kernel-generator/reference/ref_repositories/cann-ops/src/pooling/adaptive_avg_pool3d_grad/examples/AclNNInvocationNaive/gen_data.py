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


def adaptive_avgpool3d_grad_golden(grads, input):
    input_torch = torch.from_numpy(input)
    grads_torch = torch.from_numpy(grads)
    output = torch.ops.aten._adaptive_avg_pool3d_backward(
                                grads_torch,
                                input_torch
                                )
    return output.numpy()


def gen_golden_data_simple():
    dtype = np.float32
    input_shape = [2, 2, 1, 1, 4]
    grad_shape = [2, 2, 1, 1, 2]

    input = np.random.uniform(-1, 1, input_shape).astype(dtype)
    grads = np.random.uniform(-1, 1, grad_shape).astype(dtype)

    golden = adaptive_avgpool3d_grad_golden(grads, input)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input.astype(dtype).tofile("./input/input.bin")
    grads.astype(dtype).tofile("./input/grads.bin")
    golden.astype(dtype).tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

