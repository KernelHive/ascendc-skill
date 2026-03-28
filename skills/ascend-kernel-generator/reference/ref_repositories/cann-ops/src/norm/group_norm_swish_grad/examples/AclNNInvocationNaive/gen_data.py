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
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(5)

d_type_dict = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": bf16,
    'int64': np.int64
}


def do_group_norm_swish_grad(x, gamma, beta, num_groups, swish_scale, grad, mean, rstd):
    dtype = x.dtype
    if dtype == "float16" or dtype == "bfloat16":
        x = x.astype(np.float32)
        gamma = gamma.astype(np.float32)
        beta = beta.astype(np.float32)
        grad = grad.astype(np.float32)
        mean = mean.astype(np.float32)
        rstd = rstd.astype(np.float32)

    x = torch.from_numpy(x)
    gamma = torch.from_numpy(gamma)
    beta = torch.from_numpy(beta)
    grad = torch.from_numpy(grad)
    mean = torch.from_numpy(mean)
    rstd = torch.from_numpy(rstd)

    batch_num = x.size(0)
    num_channels = x.size(1)
    remaining_dims = x.size()[2:]
    hw = 1
    count = 0
    for size in remaining_dims:
        hw *= size
        count += 1
    
    num_per_group = num_channels // num_groups * hw
    if count > 1 :
        x = x.reshape((batch_num, num_channels, -1))
        grad = grad.reshape((batch_num, num_channels, -1))

    dL_dgamma = torch.zeros(num_channels)
    dL_dbeta = torch.zeros(num_channels)
    dL_dx = torch.zeros(batch_num, num_channels, hw)

    for n_i in range(batch_num):
        for g_i in range(num_groups):
            x_i = x[n_i, g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)]
            dy_i = grad[n_i, g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)]

            var_x = torch.Tensor.var(x_i, False)
            rstd_x = rstd[n_i, g_i]
            mean_x = mean[n_i, g_i]
            x_norm_i = (x_i - mean_x) * rstd_x

            gamma_i = gamma[g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)]
            beta_i = beta[g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)]

            if count == 0:
                gamma_reshape_i = gamma_i.reshape(num_channels // num_groups)
                beta_reshape_i = beta_i.reshape(num_channels // num_groups)
            else:
                gamma_reshape_i = gamma_i.reshape(num_channels // num_groups, 1)
                beta_reshape_i = beta_i.reshape(num_channels // num_groups, 1)

            dy_new_i = x_norm_i * gamma_reshape_i
            dy_new_i = dy_new_i + beta_reshape_i
            dswish_res_i = dy_new_i * (-swish_scale)
            dswish_res_i = torch.exp(dswish_res_i)
            dswish_res_i = dswish_res_i + 1.0
            tmp_res = dy_new_i / dswish_res_i
            tmp_res = dy_new_i - tmp_res
            tmp_res = tmp_res + 1.0
            dswish_res_i = tmp_res / dswish_res_i
            dy_new_i = dswish_res_i * dy_i

            if count == 0:
                temp_1 = dy_new_i
                temp_2 = dy_new_i * x_norm_i
            else:
                temp_1 = torch.sum(dy_new_i,dim = (1))
                temp_2 = torch.sum(dy_new_i * x_norm_i, dim = (1))

            dL_dbeta[g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)] += temp_1
            dL_dgamma[g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)] += temp_2
            c1 = 0
            c2 = 0
            c1 = torch.sum(temp_1 * gamma_i) / num_per_group
            c2 = torch.sum(temp_2 * gamma_i) / num_per_group

            dL_dx_G_C = torch.zeros(num_channels // num_groups, hw)

            for i in range(num_channels//num_groups):
                dL_dx_G_C[i] = rstd_x * (dy_new_i[i] * gamma[g_i * (num_channels // num_groups)+i] - x_norm_i[i] * c2 - c1)
            dL_dx[n_i, g_i * (num_channels // num_groups):(g_i+1) * (num_channels // num_groups)] = dL_dx_G_C
    dL_dx = dL_dx.numpy()
    dL_dgamma = dL_dgamma.numpy()
    dL_dbeta = dL_dbeta.numpy()

    if dtype == "float16" or dtype == "bfloat16":
        dL_dx = dL_dx.astype(d_type_dict.get(dtype))
        dL_dgamma = dL_dgamma.astype(d_type_dict.get(dtype))
        dL_dbeta = dL_dbeta.astype(d_type_dict.get(dtype))
    return dL_dx, dL_dgamma, dL_dbeta


def gen_input_data(shape, dtype, input_range, group):
    dtype = d_type_dict.get(dtype)
    dy = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    mean = np.random.uniform(input_range[0], input_range[1], (shape[1], group)).astype(dtype)
    rstd = np.random.uniform(input_range[0], input_range[1], (shape[1], group)).astype(dtype)
    x = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    gamma = np.random.uniform(input_range[0], input_range[1], shape[1]).astype(dtype)
    beta = np.random.uniform(input_range[0], input_range[1], shape[1]).astype(dtype)
    
    return dy, mean, rstd, x, gamma, beta


def gen_golden_data_simple(shape, dtype, input_range, num_groups, swish_scale):

    dy, mean, rstd, x, gamma, beta = gen_input_data(shape, dtype, input_range, group)
    golden_dx, golden_dgamma, golden_dbeta = do_group_norm_swish_grad(x, gamma, beta, num_groups, swish_scale, dy, mean, rstd)
    dy.tofile(f"./input/dy.bin")
    mean.tofile(f"./input/mean.bin")
    rstd.tofile(f"./input/rstd.bin")
    x.tofile(f"./input/x.bin")
    gamma.tofile(f"./input/gamma.bin")
    beta.tofile(f"./input/beta.bin")
    golden_dx.tofile(f"./output/golden_dx.bin")
    golden_dgamma.tofile(f"./output/golden_dgamma.bin")
    golden_dbeta.tofile(f"./output/golden_dbeta.bin")


if __name__ == "__main__":
    dtype = 'float32'
    shape, input_range = [2, 3, 4], [0.1, 1]
    group, swish_scale = 1, 1.0
    gen_golden_data_simple(shape, dtype, input_range, group, swish_scale)
