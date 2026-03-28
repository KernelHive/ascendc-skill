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
import torch
from torch import Tensor
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(5)


def do_adam_w(var, m, v, max_grad_norm, grad, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize):
    param = torch.tensor(var)
    grad = torch.tensor(grad)
    exp_avg = torch.tensor(m)
    exp_avg_sq = torch.tensor(max_grad_norm)
    max_exp_avg_sq = torch.tensor(v)
    step_t = step.item()
    param, max_exp_avg_sq, exp_avg, exp_avg_sq = single_tensor_adamw(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
        step_t, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize)

    return param.numpy(), max_exp_avg_sq.numpy(), exp_avg.numpy(), exp_avg_sq.numpy()


def single_tensor_adamw(param: Tensor,
                        grad: Tensor,
                        exp_avg: Tensor,
                        exp_avg_sq: Tensor,
                        max_exp_avg_sq: Tensor,
                        step_t,
                        amsgrad,
                        beta1,
                        beta2,
                        lr,
                        weight_decay,
                        eps,
                        maximize):
    dtype1, dtype2 = param.dtype, grad.dtype
    if dtype1 != dtype2:
        grad = grad.to(dtype1)
        max_exp_avg_sq = max_exp_avg_sq.to(dtype1)
    if dtype1 == torch.half:
        param = param.to(torch.float)
        grad = grad.to(torch.float)
        exp_avg = exp_avg.to(torch.float)
        exp_avg_sq = exp_avg_sq.to(torch.float)
        max_exp_avg_sq = max_exp_avg_sq.to(torch.float)
    if maximize:
        grad = -grad
    step_t += 1

    param.mul_(1 - lr * weight_decay)
    exp_avg.lerp_(grad, 1 - beta1)

    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step_t
    bias_correction2 = 1 - beta2 ** step_t

    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    if amsgrad:
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
    else:
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

    param = param - step_size * (exp_avg / denom)
    if dtype1 == torch.half:
        param = param.to(torch.half)
        max_exp_avg_sq = max_exp_avg_sq.to(torch.half)
        exp_avg = exp_avg.to(torch.half)
        exp_avg_sq = exp_avg_sq.to(torch.half)
    return param, max_exp_avg_sq, exp_avg, exp_avg_sq


def gen_input_data(shape, dtype, input_range, dtype2=None, step_size='float32'):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bf16,
        'int64': np.int64
    }
    if dtype2 is None:
        dtype2 = dtype
    dtype = d_type_dict.get(dtype)
    dtype2 = d_type_dict.get(dtype2)
    step_size = d_type_dict.get(step_size, np.float32)

    var = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    m = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    v = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    max_grad_norm = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype2)
    grad = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype2)

    step = np.array(np.random.randint(10)).astype(step_size)
    workspace = np.array([1])
    return var, m, v, max_grad_norm, grad, step, workspace


def gen_golden_data_simple(shape, dtype, input_range, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize,
                           dtype2=None, step_size='float32'):

    var, m, v, max_grad_norm, grad, step, workspace = gen_input_data(shape, dtype, input_range, dtype2, step_size)
    golden, _, _, _ = do_adam_w(var, m, v, max_grad_norm, grad, step, lr, beta1, beta2, weight_decay, eps,
                                amsgrad, maximize)
    var.tofile(f"./input/var.bin")
    m.tofile(f"./input/m.bin")
    v.tofile(f"./input/v.bin")
    max_grad_norm.tofile(f"./input/max_grad_norm.bin")
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
    dtype, amsgrad, maximize = sys.argv[1], sys.argv[2], sys.argv[3]

    amsgrad = parse_bool_param(sys.argv[2])
    maximize = parse_bool_param(sys.argv[3])
    if len(sys.argv) > 4:
        dtype2 = sys.argv[4]
    else:
        dtype2 = None

    if len(sys.argv) > 5:
        step_size = sys.argv[5]
    else:
        step_size = 'float32'

    shape, input_range = [2, 2, 2], [0.1, 1]
    lr, beta1, beta2, weight_decay, eps = 0.01, 0.9, 0.99, 5e-3, 1e-6
    gen_golden_data_simple(shape, dtype, input_range, lr, beta1, beta2, weight_decay, eps, amsgrad,
                           maximize, dtype2, step_size)
