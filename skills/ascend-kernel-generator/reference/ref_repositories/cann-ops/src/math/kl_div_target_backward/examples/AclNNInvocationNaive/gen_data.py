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
    dtype = np.float16
    input_shape = [4, 400]
    input1_shape = [1, 400]
    output_shape = [4, 400]
    reduction = 0
    log_target = True

    grad_output = torch.from_numpy(np.random.uniform(-1, 1, input_shape).astype(dtype))
    self_x = torch.from_numpy(np.random.uniform(-1, 1, input1_shape).astype(dtype))
    target = torch.from_numpy(np.random.uniform(-1, 1, input_shape).astype(dtype))
    if log_target:
        grad_target = target + 1
        tmp = torch.exp(target)
        grad_target = grad_target - self_x
        grad_target = grad_target * tmp
        grad_target = grad_output * grad_target
    else:
        tmp = torch.log(target)
        grad_target = tmp + 1
        grad_target = grad_target - self_x
        grad_target = grad_output * grad_target
        grad_target = grad_target.masked_fill(target == 0, 0)

    if reduction == 1:
        max_len = max(max(grad_output.numel(), self_x.numel()), target.numel())
        grad_target = grad_target / max_len

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    grad_output.numpy().astype(dtype).tofile("./input/input_x0.bin")
    self_x.numpy().astype(dtype).tofile("./input/input_x1.bin")
    target.numpy().astype(dtype).tofile("./input/input_x2.bin")
    grad_target.numpy().astype(dtype).tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

