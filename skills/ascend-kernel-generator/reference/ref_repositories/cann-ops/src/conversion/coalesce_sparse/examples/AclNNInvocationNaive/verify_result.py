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


def verify_result(real_result_indices, golden_indices, real_result_values, golden_values):
    dtype = np.float16
    real_result_values = np.fromfile(real_result_values, dtype=dtype) # 从bin文件读取实际运算结果
    golden_values = np.fromfile(golden_values, dtype=dtype) # 从bin文件读取预期运算结果
    result = np.abs(real_result_values - golden_values) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result_values), np.abs(golden_values))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, LOSS) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result_values.size * LOSS and \
           np.sum(result_atol == False) > real_result_values.size * LOSS: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] new_values result error")
            return False

    real_result_indices = np.fromfile(real_result_indices, dtype=np.int32) # 从bin文件读取实际运算结果
    golden_indices = np.fromfile(golden_indices, dtype=np.int32) # 从bin文件读取预期运算结果
    if not np.all(np.equal(real_result_indices, golden_indices)):
        print("[ERROR] new_indices result error")
        return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
