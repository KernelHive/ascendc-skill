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


def gelu_compute_erf(input_x):
    """
    compute with float32
    res = x/(1+exp(((((((a1*x^2+a2)*x^2+a3)*x^2+a4)*x^2+a5)*x^2+a6)*x^2+a7)*x))
    """
    input_x = np.maximum(input_x, -13.25)
    x1 = np.minimum(input_x, 5.75)
    x_pow = x1 * x1
    y = x_pow * (-0.3512339572e-8) + (0.2645266170e-6)  # (a1*x^2+a2)
    y = y * x_pow + (-0.7929488134e-5)                  # *x^2+a3
    y = y * x_pow + (0.1106123840e-3)                   # *x^2+a4
    y = y * x_pow + (0.6518995814e-4)                   # *x^2+a5
    y = y * x_pow + (-0.7266616915e-1)                  # *x^2+a6
    y = y * x_pow + (-0.1595769883e1)                   # *x^2+a7
    y = y * x1                                          # *x
    y = np.exp(y) + 1.0
    res = input_x / y
    return res


def tanh_parameter_compute(input_x):
    y = input_x * input_x
    y = y * input_x
    y = y * 0.044715
    result = input_x + y
    return result


def gelu_compute_tanh(input_x):
    """
    compute with float32
    gelu(x): x / (1 + exp(-sqrt(8/pi)(x + 0.044715*x^3)))
    """
    tanh_parameter = tanh_parameter_compute(input_x)    # x + 0.044715*x^3
    mul_0 = tanh_parameter * (-1.5957691)                # -sqrt(8/pi)=-1.5957691
    temp = np.exp(mul_0) + 1.0

    res = input_x / temp
    return res


def gelu(x, approximate):
    if approximate == "none":
        result = gelu_compute_erf(x)
    else:
        result = gelu_compute_tanh(x)

    return result


def round_tensor(x, round_mode):
    if round_mode == "Round":
        y = np.round(x,)
        y = np.clip(y, -128, 127)
        return y
    elif round_mode == "Floor":
        return np.floor(x)
    elif round_mode == "Ceil":
        return np.ceil(x)
    else:
        return np.trunc(x)



def ascend_quantv2(x, scale, offset):
    scale = np.broadcast_to(scale, x.shape)
    y = scale * x
    if offset is not None:
        offset = np.broadcast_to(offset, x.shape)
        y = offset + y

    y = round_tensor(y, "Round")
    y = torch.from_numpy(y).to(torch.int8)

    return y


def dynamic_quant(x, scale):
    if scale is not None:
        scale = np.broadcast_to(scale, x.shape)
        mul_res = x * scale
    else:
        mul_res = x
    abs_res = np.abs(mul_res)
    max_res = np.max(abs_res, axis=-1, keepdims=True)
    max_res_reshape = np.max(abs_res, axis=-1, keepdims=False)
    tmp_out_scale = 127.0 / max_res
    out_scale = 1 / tmp_out_scale
    out_scale = np.reshape(out_scale, max_res_reshape.shape)
    tmp_out_scale = np.broadcast_to(tmp_out_scale, x.shape)
    y = mul_res * tmp_out_scale
    y = round_tensor(y, "Round")
    y = torch.from_numpy(y).to(torch.int8)
    out_scale = torch.from_numpy(out_scale)
    print("++++++++++out_scale", out_scale.shape)
    print("++++++++++out_scale", out_scale.dtype)
    return y, out_scale


def gelu_quant_golden(x, scale, offset, approximate, quant_mode):

    # 计算前转为fp32
    if offset is not None:
        scale = scale
    if offset is not None:
        offset = offset

    gelu_res = gelu(x, approximate)
    if quant_mode == 'static':
        y = ascend_quantv2(gelu_res, scale, offset)
        return y
    else:
        y, out_scale = dynamic_quant(gelu_res, scale)
        return y, out_scale

def gen_golden_data_simple():
    dtype = np.float32
    input_shape = [1, 2, 4]
    inputscale_shape = [4]
    inputoffset_shape = [4]

    x = np.random.uniform(-1, 1, input_shape).astype(dtype)
    scale = np.random.uniform(-1, 1, inputscale_shape).astype(dtype)
    offset = np.random.uniform(-1, 1, inputoffset_shape).astype(dtype)
    golden, out_scale = gelu_quant_golden(x, scale, offset, "none", "dynamic")

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_x.bin")
    scale.astype(dtype).tofile("./input/input_scale.bin")
    offset.astype(dtype).tofile("./input/input_offset.bin")
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

