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

LOSS = 0


def verify_result(real_result_path, golden_path):
    dtype = np.int32
    real = np.fromfile(real_result_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if len(real) != len(golden):
        print(f"[ERROR] Length mismatch: real has {len(real)}, golden has {len(golden)}")
        return False

    diff = np.abs(real - golden)
    mismatch = np.where(diff > LOSS)[0]

    if len(mismatch) > 0:
        print("[ERROR] result error")
        print("First 10 mismatches:")
        for i in range(min(10, len(mismatch))):
            idx = mismatch[i]
            r_val = real[idx]
            g_val = golden[idx]
            print(f"Index: {idx}, Real: {r_val}, Golden: {g_val}")
        return False

    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2])
