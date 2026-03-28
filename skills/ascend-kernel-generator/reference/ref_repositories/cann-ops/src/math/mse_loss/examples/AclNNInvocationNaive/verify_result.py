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

LOSS = 1e-4


def verify_result(real_result, golden):
    result = np.fromfile(real_result, dtype=np.float32)
    golden = np.fromfile(golden, dtype=np.float32)
    for i, (res, gold) in enumerate(zip(result, golden)):
        diff = abs(res - gold)
        if (diff > LOSS) and (diff / gold > LOSS):
            error_message = f"output[{i}] is {res}, expect {gold}"
            print(error_message)
            return False
    
    print("test pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2])