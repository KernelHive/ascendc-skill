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
import torch
import numpy as np

def gen_golden_data_simple():
    dtype = np.float16
    self_shape = [1, 2, 3]
    mat2_shape = [1, 3, 4]
    output_shape = [1, 2, 4]

    self = np.random.uniform(-1, 1, self_shape).astype(dtype)
    mat2 = np.random.uniform(-1, 1, mat2_shape).astype(dtype)

    golden = []
    batch = self_shape[0]
    for i in range(batch):
        out = self[i] @ mat2[i]
        golden.append(out)

    golden = np.stack(golden, axis=0)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    self.astype(dtype).tofile("./input/input_self.bin")
    mat2.astype(dtype).tofile("./input/input_mat2.bin")
    golden.astype(dtype).tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


