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
import torch
import numpy as np


def instance_norm_golden(x, gamma, beta, eps, layout):
    ori_dtype = x.dtype
    if ori_dtype != torch.float32:
        x = x.type(torch.float32)
        gamma = gamma.type(torch.float32)
        beta = beta.type(torch.float32)
    reduce_axis = [1, 2] if layout == 'NHWC' else [2, 3]
    gamma = gamma.reshape([1, 1, 1, gamma.shape[0]]) if layout == 'NHWC' else gamma.reshape([1, gamma.shape[0], 1, 1])
    beta = beta.reshape([1, 1, 1, beta.shape[0]]) if layout == 'NHWC' else beta.reshape([1, beta.shape[0], 1, 1])
    mean = torch.mean(x, dim=reduce_axis, keepdim=True)
    var = torch.mean(torch.pow((x - mean), 2), dim=reduce_axis, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    tmp_x = (x - mean) * rstd
    y = tmp_x * gamma + beta
    if ori_dtype != torch.float32:
        return y.type(ori_dtype), mean, var
    else:
        return y, mean, var


def gen_golden_data_simple():
    input_x = torch.ones(128, dtype=torch.float32).reshape(1, 8, 4, 4) * 0.77
    input_gamma = torch.ones(8, dtype=torch.float32).reshape(8) * 1.5
    input_beta = torch.ones(8, dtype=torch.float32).reshape(8) * 0.5
    epsilon = 1e-5
    layout = "NCHW"

    res, mean, variance = instance_norm_golden(input_x, input_gamma, input_beta, epsilon, layout)

    res.numpy().tofile("./output/golden1.bin")
    mean.numpy().tofile("./output/golden2.bin")
    variance.numpy().tofile("./output/golden3.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

