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


def gen_golden_data_simple(bias_flag):
    x1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32).reshape(1, 2, 8)
    x2 = np.array([4, 4, 4, 4, 4, 4, 4, 4, -3, -3, -3, -3, -3, -3, -3, -3], dtype=np.float32).reshape(1, 2, 8)
    gamma = np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32).reshape(8)
    beta = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32).reshape(8)
    bias = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32).reshape(8)
    epsilon = np.float32(1e-06)

    if bias_flag:
        x = x1 + x2 + bias
    else:
        x = x1 + x2

    input_shape = x1.shape
    row_size = x1.shape[-1]
    row_count = 1
    for i in range(0, len(input_shape) - 1):
        row_count *= input_shape[i]

    x_shape = (row_count, row_size)
    x_mean_shape = (row_count, 1)

    x = x.reshape(x_shape)
    x_mean = np.mean(x, axis=1).reshape(x_mean_shape)
    x_var = np.var(x, axis=1).reshape(x_mean_shape) + epsilon
    x_rstd = 1.0 / np.sqrt(x_var)

    x_mean_broadcast = np.broadcast_to(x_mean, x_shape)
    x_rstd_broadcast = np.broadcast_to(x_rstd, x_shape)
    gamma_broadcast = np.broadcast_to(gamma, x_shape)
    beta_broadcast = np.broadcast_to(beta, x_shape)

    y = np.multiply(np.multiply(x - x_mean_broadcast, x_rstd_broadcast), gamma_broadcast) + beta_broadcast

    if bias_flag:
        y.tofile("./output/goldeny_bias.bin")
        x_mean.tofile("./output/goldenmean_bias.bin")
        x_rstd.tofile("./output/goldenrstd_bias.bin")
        x.tofile("./output/goldenx_bias.bin")
    else:
        y.tofile("./output/goldeny.bin")
        x_mean.tofile("./output/goldenmean.bin")
        x_rstd.tofile("./output/goldenrstd.bin")
        x.tofile("./output/goldenx.bin")

if __name__ == "__main__":
    gen_golden_data_simple(True)
    gen_golden_data_simple(False)

