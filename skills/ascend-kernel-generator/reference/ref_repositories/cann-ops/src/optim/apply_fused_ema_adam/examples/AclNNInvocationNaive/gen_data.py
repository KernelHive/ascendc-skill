#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================

import sys
import os
import math
import numpy as np
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(5)

d_type_dict = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": bf16,
    'int64': np.int64
}


def correction_compute(bias_correction, beta1, beta2, step):
    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction:
        bias_correction1 = 1 - math.pow(beta1, step)
        bias_correction2 = 1 - math.pow(beta2, step)
    return bias_correction1, bias_correction2


def do_adam_ema(grad, var, m, v, s, step, lr, beta1, beta2, weight_decay, eps, mode, bias, ema_decay, dtype):
    if dtype == "float16" or dtype == "bfloat16":
        grad = grad.astype(np.float32)
        var = var.astype(np.float32)
        m = m.astype(np.float32)
        v = v.astype(np.float32)
        s = s.astype(np.float32)
    beta1_correction, beta2_correction = correction_compute(bias, beta1, beta2, step)
    if mode == 0:
        grad_ = grad + weight_decay * var
    elif mode == 1:
        grad_ = grad
    m_ = beta1 * m + (1 - beta1) * grad_
    v_ = beta2 * v + (1 - beta2) * grad_ * grad_
    next_m = m_ / beta1_correction
    next_v = v_ / beta2_correction
    denom = np.sqrt(next_v) + eps
    if mode == 0:
        update = next_m / denom
    elif mode == 1:
        update = next_m / denom + weight_decay * var
    var_ = var - lr * update
    s_ = ema_decay * s + (1 - ema_decay) * var_
    if dtype == "float16" or dtype == "bfloat16":
        var_ = var_.astype(d_type_dict.get(dtype))
        m_ = m_.astype(d_type_dict.get(dtype))
        v_ = v_.astype(d_type_dict.get(dtype))
        s_ = s_.astype(d_type_dict.get(dtype))
    return [var_, m_, v_, s_]


def gen_input_data(shape, dtype, input_range, step_size='int64'):
    dtype2 = dtype
    dtype = d_type_dict.get(dtype)
    dtype2 = d_type_dict.get(dtype2)
    step_size = d_type_dict.get(step_size, np.int64)

    var = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    m = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    v = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    s = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype2)
    grad = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype2)

    step = np.array(np.random.randint(10)).astype(step_size)
    workspace = np.array([1])
    return var, m, v, s, grad, step, workspace


def gen_golden_data_simple(shape, dtype, input_range, lr, beta1, beta2, weight_decay, eps, mode, bias, ema_decay,
                           step_size="int64"):

    var, m, v, s, grad, step, workspace = gen_input_data(shape, dtype, input_range, step_size)
    golden, _, _, _ = do_adam_ema(grad, var, m, v, s, step, lr, beta1, beta2, weight_decay, eps, mode, bias,
                                  ema_decay, dtype)
    var.tofile(f"./input/var.bin")
    m.tofile(f"./input/m.bin")
    v.tofile(f"./input/v.bin")
    s.tofile(f"./input/s.bin")
    grad.tofile(f"./input/grad.bin")
    step.tofile(f"./input/step.bin")
    workspace.tofile(f"./input/workspace.bin")
    golden.tofile(f"./output/golden.bin")


def parse_bool_param(param):
    if param.lower() in ['true', 't']:
        return True
    return False


if __name__ == "__main__":
    # 清理bin文件
    dtype, bias = sys.argv[1], sys.argv[2]
    bias = parse_bool_param(bias)
    step_size = "int64"

    shape, input_range = [2, 2, 2], [0.1, 1]
    lr, beta1, beta2, weight_decay, eps, mode, ema_decay = 0.001, 0.9, 0.999, 0.5, 1e-8, 1, 0.5
    gen_golden_data_simple(shape, dtype, input_range, lr, beta1, beta2, weight_decay, eps, mode, bias,
                           ema_decay, step_size)
