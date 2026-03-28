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

MINIMUM = 10e-10


def verify_result(dtype, real_result, golden):
    acc = 0.0009765625
    if dtype == "float16":
        acc = 0.001953125
    elif dtype == "bfloat16":
        acc = 0.015625
    elif dtype == "float32":
        acc = 0.0009765625
    real_result = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    result = np.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), acc) # 计算相对误差
    if not result_rtol.all():
        if np.sum(result_rtol == False) > real_result.size * acc: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] real_result = ", real_result)
            print("[ERROR] golden = ", golden)
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2], sys.argv[3])
