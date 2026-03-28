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
    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3, 1, 4)
    input_gx = np.array([2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8], dtype=np.float32).reshape(3, 1, 4)
    input_beta = np.array([0, 1, 2, 3], dtype=np.float32).reshape(4)
    input_gamma = np.array([0, 1, 2, 3], dtype=np.float32).reshape(4)
    alpha = 0.3
    eps = 1e-06

    dtype = input_x.dtype
    input_shape = input_x.shape
    reduce_axis = -1

    if "float16" in str(dtype):
        input_x = input_x.astype(np.float32)
        input_gx = input_gx.astype(np.float32)
        input_beta = input_beta.astype(np.float32)
        input_gamma = input_gamma.astype(np.float32)

    x = torch.from_numpy(input_x)
    gx = torch.from_numpy(input_gx)
    beta = torch.from_numpy(input_beta)
    gamma = torch.from_numpy(input_gamma)

    x_add = x * alpha + gx
    mean = x_add.mean(-1, keepdim=True)
    diff = x_add - mean
    variance = diff.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    res = gamma * diff * rstd + beta

    res = res.numpy()
    if "float16" in str(dtype):
        res = res.astype(dtype)
    else:
        res = res.astype(np.float32)
    mean = mean.numpy().astype(np.float32)
    rstd = rstd.numpy().astype(np.float32)

    mean.tofile("./output/golden_mean.bin")
    rstd.tofile("./output/golden_rstd.bin")
    res.tofile("./output/golden_y.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

