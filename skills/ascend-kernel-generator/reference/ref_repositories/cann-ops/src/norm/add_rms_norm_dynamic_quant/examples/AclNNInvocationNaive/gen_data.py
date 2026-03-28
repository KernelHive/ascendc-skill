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


def goldenAddRmsNorm(x1, x2, gamma, eps):
    ori_dtype = x1.dtype
    if ori_dtype != torch.float32:
        x1 = x1.type(torch.float32)
        x2 = x2.type(torch.float32)
        gamma = gamma.type(torch.float32)
    x = x1 + x2
    rstd = torch.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + eps)
    y = x * rstd * gamma
    if ori_dtype != torch.float32:
        return y, x.type(ori_dtype)
    else:
        return y, x


def goldenDynamicQuant(x, smooth):
    x = x if x.dtype == torch.float32 else x.type(torch.float32)
    if (smooth is not None):
        smooth = smooth if smooth.dtype == torch.float32 else smooth.type(torch.float32)
    else:
        smooth = None
    smooth_x = x if (smooth is None) else x * smooth
    x_max = torch.max(torch.abs(smooth_x), axis=-1, keepdim=True)[0]
    gs_rev = 127.0 / x_max
    gs = 1 / gs_rev
    sx = smooth_x * gs_rev
    gq = torch.round(sx).type(torch.int8)
    return gq, gs


def gen_golden_data_simple():

    x1 = np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], dtype=np.float16).reshape(2, 8)
    x2 = np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], dtype=np.float16).reshape(2, 8)
    gamma = np.array([7, 7, 7, 7, 7, 7, 7, 7], dtype=np.float16).reshape(8)
    smooth1 = np.array([7, 7, 7, 7, 7, 7, 7, 7], dtype=np.float16).reshape(8)
    smooth2 = np.array([7, 7, 7, 7, 7, 7, 7, 7], dtype=np.float16).reshape(8)

    epsilon = 1e-06

    smooth1Exist = smooth1 is not None
    smooth2Exist = smooth2 is not None

    dtype_real = x1.dtype

    if 'bf16' in str(dtype_real) or 'bfloat16' in str(dtype_real):
        tensor_x1 = torch.from_numpy(x1.view(np.int16)).view(torch.bfloat16)
        tensor_x2 = torch.from_numpy(x2.view(np.int16)).view(torch.bfloat16)
        tensor_gamma = torch.from_numpy(gamma.view(np.int16)).view(torch.bfloat16)
        tensor_smooth1 = torch.from_numpy(smooth1.view(np.int16)).view(torch.bfloat16) if smooth1Exist else None
        tensor_smooth2 = torch.from_numpy(smooth2.view(np.int16)).view(torch.bfloat16) if smooth2Exist else None
    else:
        tensor_x1 = torch.from_numpy(x1)
        tensor_x2 = torch.from_numpy(x2)
        tensor_gamma = torch.from_numpy(gamma)
        tensor_smooth1 = torch.from_numpy(smooth1) if smooth1Exist else None
        tensor_smooth2 = torch.from_numpy(smooth2) if smooth2Exist else None

    gyFp32, gx = goldenAddRmsNorm(tensor_x1, tensor_x2, tensor_gamma, epsilon)
    if smooth1Exist and smooth2Exist:
        gq1, gs1 = goldenDynamicQuant(gyFp32, tensor_smooth1)
        gq2, gs2 = goldenDynamicQuant(gyFp32, tensor_smooth2)
    elif smooth1Exist and (not smooth2Exist):
        gq1, gs1 = goldenDynamicQuant(gyFp32, tensor_smooth1)
        gq2, gs2 = torch.zeros(gq1.shape, dtype=gq1.dtype), torch.zeros(gs1.shape, dtype=gs1.dtype)
    elif (not smooth2Exist) and (not smooth2Exist):
        gq1, gs1 = goldenDynamicQuant(gyFp32, None)
        gq2, gs2 = torch.zeros(gq1.shape, dtype=gq1.dtype), torch.zeros(gs1.shape, dtype=gs1.dtype)
    else:
        pass


    gq1_np = gq1.numpy()
    gq2_np = gq2.numpy()
    gs1_np = gs1.numpy()
    gs2_np = gs2.numpy()

    if 'bf16' in str(dtype_real) or 'bfloat16' in str(dtype_real):
        x_np = gx.view(torch.int16).numpy().view(dtype_real)
    else:
        x_np = gx.numpy()

    gq1_np.tofile("./output/goldeny.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

