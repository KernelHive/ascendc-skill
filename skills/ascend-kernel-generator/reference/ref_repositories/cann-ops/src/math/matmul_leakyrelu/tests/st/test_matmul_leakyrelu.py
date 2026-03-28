#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import numpy as np


def matmul_leakyrelu_test(a, b, bias):
    alpha = 0.001
    golden = (np.matmul(a.astype(np.float32), b.astype(np.float32)) + bias).astype(np.float32)
    res = np.where(golden >= 0, golden, golden * alpha)
    return res


def calc_expect_func(a, b, bias, y):
    res = matmul_leakyrelu_test(a["value"], b["value"], bias["value"])
    return [res]