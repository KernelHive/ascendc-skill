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

LOSS = 1e-4         # 容忍偏差
MINIMUM = 10e-10


def verify_result(real_result_y, real_result_indices, golden_y, golden_indices):
    real_result_y = np.fromfile(real_result_y, dtype=np.float32)
    golden_y = np.fromfile(golden_y, dtype=np.float32)
    real_result_indices = np.fromfile(real_result_indices, dtype=np.int32)
    golden_indices = np.fromfile(golden_indices, dtype=np.int32)

    result_y = np.abs(real_result_y - golden_y)
    deno_y = np.maximum(np.abs(real_result_y), np.abs(golden_y))
    result_atol_y = np.less_equal(result_y, LOSS)
    result_rtol_y = np.less_equal(result_y / np.add(deno_y, MINIMUM), LOSS)

    result_indices = np.abs(real_result_indices - golden_indices)
    deno_indices = np.maximum(np.abs(real_result_indices), np.abs(golden_indices))
    result_atol_indices = np.less_equal(result_indices, LOSS)
    result_rtol_indices = np.less_equal(result_indices / np.add(deno_y, MINIMUM), LOSS)

    if not result_rtol_y.all() and not result_atol_y.all():
        if np.sum(result_rtol_y == False) > real_result_y.size * LOSS and \
           np.sum(result_atol_y == False) > real_result_y.size * LOSS:
            print("[ERROR] result_y error")
            return False

    if not result_rtol_indices.all() and not result_atol_indices.all():
        if np.sum(result_rtol_y == False) > real_result_y.size * LOSS and \
           np.sum(result_atol_y == False) > real_result_y.size * LOSS:
            print("[ERROR] result_y error")
            return False           
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
