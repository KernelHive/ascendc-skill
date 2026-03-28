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

    x1 = np.array([1,2,3,4,5,6]).astype(np_dtype)
    x2 = np.array([7,8,9]).astype(np_dtype)
    y1 = np.array([4,3,8,9,3,5]).astype(np_dtype)
    y2 = np.array([5,6,7]).astype(np_dtype)
    z1 = np.array([1,2,3,4,5,6]).astype(np_dtype)
    z2 = np.array([7,8,9]).astype(np_dtype)
    alpha1 = 1.2
    alpha2 = 2.2

    out1 = x1 + y1 * z1 * alpha1
    out2 = x2 + y2 * z2 * alpha2


    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1.astype(np_dtype).tofile("./input/input_x1.bin")
    x2.astype(np_dtype).tofile("./input/input_x2.bin")
    y1.astype(np_dtype).tofile("./input/input_y1.bin")
    y2.astype(np_dtype).tofile("./input/input_y2.bin")
    z1.astype(np_dtype).tofile("./input/input_z1.bin")
    z2.astype(np_dtype).tofile("./input/input_z2.bin")
    out1.astype(np_dtype).tofile("./output/golden_out1.bin")
    out2.astype(np_dtype).tofile("./output/golden_out2.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

