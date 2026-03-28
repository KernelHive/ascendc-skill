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
import subprocess
import numpy as np


def add_rms_golden_milan(x1, x2, gamma, eps=1e-6):
    # milan上的实现，不同分支走的cast方案不同，但ttk感知不敏感，已目前标杆为准，aclnn框架会针对适配
    if x1.dtype == torch.bfloat16:
        x = (x1.type(torch.float32) + x2.type(torch.float32)).type(x1.dtype)
    else:
        x = x1 + x2

    xFp32 = x.type(torch.float32)
    rstd = torch.rsqrt(xFp32.pow(2).mean(-1, keepdim=True) + eps)
    tmpX = xFp32 * rstd

    if x1.dtype == torch.bfloat16:
        tmpX = tmpX.type(torch.bfloat16).type(torch.float32)
        y = (tmpX * gamma.type(torch.float32)).type(x1.dtype)
    elif x.dtype == torch.float16:
        tmpX = tmpX.type(torch.float16)
        y = tmpX * gamma
    else:
        y = tmpX * gamma
    return y, rstd, x


def add_rms_golden_torino(x1, x2, gamma, eps=1e-6):
    x = (x1.type(torch.float32) + x2.type(torch.float32))

    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    tmpX = x * rstd

    y = (tmpX * gamma.type(torch.float32)).type(x1.dtype)
    x = x.type(x1.dtype)
    return y, rstd, x

def get_soc_version():
    # 获取原始输出
    raw_output = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    lines = raw_output.strip().split("\n")
    if len(lines) >= 2:
        second_line = lines[1]
        fields = second_line.split()
        if len(fields) >= 5:
            result = fields[3] + fields[4]
            return result
        else:
            return None
    else:
        return None 



def gen_golden_data_simple():
    x1_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2,
                       3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(2, 16)
    x2_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2,
                       3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(2, 16)
    gamma_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(16)
    input_dtype = x1_data.dtype
    if x1_data.dtype == np.float16:
        dtype = "fp16"
    elif x1_data.dtype == np.float32:
        dtype = "fp32"
    else:
        dtype = "bf16"

    if dtype == 'bf16':
        x1_tensor = torch.tensor(x1_data.astype(np.float32)).to(torch.bfloat16)
        x2_tensor = torch.tensor(x2_data.astype(np.float32)).to(torch.bfloat16)
        gamma_tensor = torch.tensor(gamma_data.astype(np.float32)).to(torch.bfloat16)
    else:
        x1_tensor = torch.tensor(x1_data)
        x2_tensor = torch.tensor(x2_data)
        gamma_tensor = torch.tensor(gamma_data)
    epsilon = 1e-6

    short_soc_version = get_soc_version()
    if short_soc_version is not None and (("Ascend910B" in short_soc_version) or ("Ascend910_93" in short_soc_version)):
        y_tensor, var_tensor, x_tensor = add_rms_golden_milan(x1_tensor, x2_tensor, gamma_tensor, eps=epsilon)
    else:
        y_tensor, var_tensor, x_tensor = add_rms_golden_torino(x1_tensor, x2_tensor, gamma_tensor, eps=epsilon)

    if dtype == 'bf16':
        y = y_tensor.to(torch.float32).numpy().astype(input_dtype)
        x = x_tensor.to(torch.float32).numpy().astype(input_dtype)
    else:
        y = y_tensor.numpy()
        x = x_tensor.numpy()
    rstd = var_tensor.numpy()

    y.tofile("./output/goldeny.bin")
    rstd.tofile("./output/goldenrstd.bin")
    x.tofile("./output/goldenx.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

