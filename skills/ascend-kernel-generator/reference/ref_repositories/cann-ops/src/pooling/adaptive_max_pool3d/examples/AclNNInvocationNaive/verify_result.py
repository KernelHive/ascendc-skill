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
import sys
import numpy as np

LOSS = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
MINIMUM = 10e-10

def verify_result(real_result_out, real_result_indices, golden_out, golden_indices):
    dtype = np.float32
    real_result_out = np.fromfile(real_result_out, dtype=dtype) # 从bin文件读取实际运算结果
    real_result_indices = np.fromfile(real_result_indices, dtype=np.int32) # 从bin文件读取实际运算结果

    golden_out = np.fromfile(golden_out, dtype=dtype) # 从bin文件读取预期运算结果
    golden_indices = np.fromfile(golden_indices, dtype=np.int64) # 从bin文件读取预期运算结果

    result_out = np.abs(real_result_out - golden_out) # 计算运算结果和预期结果偏差
    result_indices = np.abs(real_result_indices - golden_indices) # 计算运算结果和预期结果偏差

    deno_out = np.maximum(np.abs(real_result_out), np.abs(golden_out))  # 获取最大值并组成新数组
    deno_indices = np.maximum(np.abs(real_result_indices), np.abs(golden_indices))  # 获取最大值并组成新数组

    result_out_atol = np.less_equal(result_out, LOSS) # 计算绝对误差
    result_indices_atol = np.less_equal(result_indices, LOSS) # 计算绝对误差

    result_out_rtol = np.less_equal(result_out / np.add(deno_out, MINIMUM), LOSS) # 计算相对误差
    result_indices_rtol = np.less_equal(result_indices / np.add(deno_indices, MINIMUM), LOSS) # 计算相对误差

    if not result_out_rtol.all() and not result_out_atol.all():
        if np.sum(result_out_rtol == False) > real_result_out.size * LOSS and \
           np.sum(result_out_atol == False) > real_result_out.size * LOSS: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False

    if not result_indices_rtol.all() and not result_indices_atol.all():
        if np.sum(result_indices_rtol == False) > real_result_indices.size * LOSS and \
           np.sum(result_indices_atol == False) > real_result_indices.size * LOSS: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
