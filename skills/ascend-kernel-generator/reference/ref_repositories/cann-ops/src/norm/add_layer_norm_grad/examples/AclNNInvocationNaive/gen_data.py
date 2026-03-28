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
    input_dy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3, 1, 4)
    input_x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3, 1, 4)
    input_x2 = np.array([2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8], dtype=np.float32).reshape(3, 1, 4)
    input_rstd = np.array([0, 1, 2], dtype=np.float32).reshape(3, 1, 1)
    input_mean = np.array([0, 1, 2], dtype=np.float32).reshape(3, 1, 1)
    input_gamma = np.array([0, 1, 2, 3], dtype=np.float32)
    input_dsum = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3, 1, 4)

    input_shape = input_dy.shape
    reduce_axis = -1

    dy_hp = input_dy.astype(np.float32)
    rstd_hp = input_rstd.astype(np.float32)
    mean_hp = input_mean.astype(np.float32)
    gamma_hp = input_gamma.astype(np.float32)

    x_hp = input_x1.astype(np.float32) + input_x2.astype(np.float32)
    d = input_shape[-1]
    axis = tuple([_ for _ in range(len(input_shape) - 1)])

    pd_xl = dy_hp * gamma_hp
    x2_tensor = x_hp - mean_hp

    pd_var_first_part = (-0.5) * pd_xl * x2_tensor * np.power(rstd_hp, 3)
    pd_var = np.sum(pd_var_first_part, reduce_axis, keepdims=True)

    pd_mean_first_part = np.sum(((-1.0) * pd_xl * rstd_hp), reduce_axis, keepdims=True)
    pd_mean_second_part = np.sum(((-2.0) * x2_tensor), reduce_axis, keepdims=True)
    pd_mean = pd_mean_first_part + pd_var * (1.0 / d) * pd_mean_second_part

    pd_x_first_part = pd_xl * rstd_hp
    pd_x_second_part = pd_var * (2.0 / d) * x2_tensor + pd_mean * (1.0 / d)
    golden_x = pd_x_first_part + pd_x_second_part
    golden_x = pd_x_first_part
    golden_x += input_dsum

    golden_gamma = np.sum(dy_hp * x2_tensor * rstd_hp, axis=axis)
    golden_beta = np.sum(dy_hp, axis=axis)

    golden_x.tofile("./output/golden1.bin")
    golden_gamma.tofile("./output/golden2.bin")
    golden_beta.tofile("./output/golden3.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

