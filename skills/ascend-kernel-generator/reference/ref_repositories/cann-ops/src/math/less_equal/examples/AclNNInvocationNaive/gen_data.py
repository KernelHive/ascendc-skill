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

import numpy as np
import os

def gen_golden_data_simple():
    input_x1 = np.random.uniform(-10, 10, [1024, 1024]).astype(np.float32)
    input_x2 = np.random.uniform(-10, 10, [1024, 1024]).astype(np.float32)
    golden = np.less_equal(input_x1, input_x2)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
