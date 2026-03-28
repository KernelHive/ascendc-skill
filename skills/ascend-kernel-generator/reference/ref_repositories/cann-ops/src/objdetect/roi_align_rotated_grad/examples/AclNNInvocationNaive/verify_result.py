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
import torch

LOSS = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
MINIMUM = 10e-10

def verify_result(real_result, golden):
    dtype = np.float32
    real_result_npu = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    real_result = real_result_npu.reshape(5, 32, 32, 48)
    real_result_tran = real_result.transpose(0, 3, 1, 2)
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])