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
    input_x1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0], dtype=np.float16).reshape(2, 16)
    input_x2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0], dtype=np.float16).reshape(2, 16)
    input_gamma = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16).reshape(16)
    input_scales1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float16).reshape(16)
    input_scales2 = None
    input_zero_points1 = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                  100, 100, 100, 100], dtype=np.float16).reshape(16)
    input_zero_points2 = None

    epsilon = 1e-6
    divMode = True
    n = len(input_x1.shape) - len(input_gamma.shape)
    input_gamma = input_gamma.reshape(np.multiply.reduce(np.array(input_gamma.shape)))
    input_scales1 = input_scales1.reshape(np.multiply.reduce(np.array(input_scales1.shape)))
    if input_scales2 is not None:
        input_scales2 = input_scales2.reshape(np.multiply.reduce(np.array(input_scales2.shape)))
    if input_zero_points1 is not None:
        input_zero_points1 = input_zero_points1.reshape(np.multiply.reduce(np.array(input_zero_points1.shape)))
    if input_zero_points2 is not None:
        input_zero_points2 = input_zero_points2.reshape(np.multiply.reduce(np.array(input_zero_points2.shape)))
    x1_shape = input_x1.shape[0:n] + input_gamma.shape
    input_x1 = input_x1.reshape(x1_shape)
    input_x2 = input_x2.reshape(x1_shape)

    len_shape_x = len(input_x1.shape)
    len_shape_gamma = len(input_gamma.shape)
    axis = len_shape_x - len_shape_gamma
    input_x_dtype = input_x1.dtype
    input_scales_dtype = input_scales1.dtype
    if input_zero_points1 is not None:
        input_zp_dtype = input_zero_points1.dtype
    if input_zero_points2 is not None:
        input_zp_dtype = input_zero_points2.dtype

    if (input_x_dtype == np.float32):
        add_x = input_x1 + input_x2
    elif (input_x_dtype == np.float16):
        add_x = (input_x1.astype(np.float32) + input_x2.astype(np.float32))
    else:
        add_x = (input_x1.astype(np.float32) + input_x2.astype(np.float32))

    if input_scales_dtype is not np.float32:
        input_scales1 = input_scales1.astype(np.float32)
    if input_scales2 is not None:
        if input_scales_dtype is not np.float32:
            input_scales2 = input_scales2.astype(np.float32)
    if input_zero_points1 is not None:
        if input_zp_dtype == np.int32:
            input_zero_points1 = input_zero_points1.astype(np.float32)
        elif input_scales_dtype is not np.float32:
            input_zero_points1 = input_zero_points1.astype(np.float32)
    if input_zero_points2 is not None:
        if input_zp_dtype == np.int32:
            input_zero_points2 = input_zero_points2.astype(np.float32)
        elif input_scales_dtype is not np.float32:
            input_zero_points2 = input_zero_points2.astype(np.float32)

    x_fp32 = add_x.astype(np.float32)
    variance = np.mean(np.power(x_fp32, 2), axis=axis, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd = 1 / std
    result_mid = x_fp32 * rstd

    if input_x_dtype == np.float32:
        y_array = result_mid * input_gamma
    elif input_x_dtype == np.float16:
        input_gamma_fp32 = input_gamma.astype(np.float32)
        y_array = result_mid * input_gamma_fp32
    else:
        input_gamma_fp32 = input_gamma.astype(np.float32)
        y_array = result_mid * input_gamma_fp32

    tensor_scales1 = torch.from_numpy(input_scales1)
    tensor_scales1 = tensor_scales1.to(torch.float32)


    if input_zero_points1 is None:
        tensor_zero_points1 = torch.zeros(input_scales1.shape, dtype=torch.float32)
    else:
        tensor_zero_points1 = torch.from_numpy(input_zero_points1)
    if input_scales2 is None:
        tensor_scales2 = torch.ones(input_scales1.shape, dtype=torch.float32)
    else:
        tensor_scales2 = torch.from_numpy(input_scales2)
    if input_zero_points2 is None:
        tensor_zero_points2 = torch.zeros(input_scales1.shape, dtype=torch.float32)
    else:
        tensor_zero_points2 = torch.from_numpy(input_zero_points2)

    if not divMode:
        tensor_scales1 = 1.0 / tensor_scales1
        if input_scales2 is not None:
            tensor_scales2 = 1.0 / tensor_scales2

    y = torch.from_numpy(y_array).type(torch.float32)
    y1 = torch.quantize_per_channel(y, tensor_scales1, tensor_zero_points1, axis, torch.qint8)
    y1_np = y1.int_repr().detach().clone().cpu().numpy()

    y2 = torch.quantize_per_channel(y, tensor_scales2, tensor_zero_points2, axis, torch.qint8)
    y2_np = y2.int_repr().detach().clone().cpu().numpy()

    y1_np.reshape(input_x1.shape).tofile("./output/goldeny1.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

