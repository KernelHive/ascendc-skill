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
import numpy as np

def gen_golden_data_simple():
    np_dtype = np.float32
    input_shape = [8, 2048]

    x1 = np.random.uniform(-1, 1, input_shape).astype(np_dtype)
    x2 = np.random.uniform(-1, 1, input_shape).astype(np_dtype)

    out1 = x1
    out2 = x2

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1.astype(np_dtype).tofile("./input/input_x1.bin")
    x2.astype(np_dtype).tofile("./input/input_x2.bin")
    out1.astype(np_dtype).tofile("./output/golden_out1.bin")
    out2.astype(np_dtype).tofile("./output/golden_out2.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

