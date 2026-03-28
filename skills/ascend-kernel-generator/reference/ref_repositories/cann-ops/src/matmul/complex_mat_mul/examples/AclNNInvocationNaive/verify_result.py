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

import sys
import numpy as np

LOSS = 1e-3
MINIMUM = 10e-10


def verify_result(actual, golden):
    dtype = np.complex64
    actual = np.fromfile(actual, dtype=dtype)
    golden = np.fromfile(golden, dtype=dtype)
    
    # Calculate absolute and relative errors
    abs_diff = np.abs(actual - golden)
    max_vals = np.maximum(np.abs(actual), np.abs(golden))
    
    abs_ok = np.all(abs_diff <= LOSS)
    rel_ok = np.all(abs_diff / (max_vals + MINIMUM) <= LOSS)
    
    if abs_ok and rel_ok:
        print("test pass")
        return True
    else:
        error_count = np.sum((abs_diff > LOSS) & (abs_diff / (max_vals + MINIMUM) > LOSS))
        total = actual.size
        error_percent = error_count / total
        if error_percent > LOSS:
            print(f"[ERROR] result error: {error_count}/{total} elements failed")
            return False
        print("test pass")
        return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: verify_result.py <actual> <golden>")
        sys.exit(1)
    verify_result(sys.argv[1], sys.argv[2])
