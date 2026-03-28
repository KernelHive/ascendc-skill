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
    dtype = np.float32
    idx_dtype = np.int64
    varRefShape = [5, 3, 4]
    indiceShape = [1, 2]
    updatesShape = [1, 5, 3, 4]

    varOne_tensor = torch.randn(varRefShape).to(torch.float32)
    indice_tensor = torch.randint(0, 2, indiceShape).to(torch.int64)
    updates_tensor = torch.randn(updatesShape).to(torch.float32)
    var = [varOne_tensor]

    varOne_numpy = varOne_tensor.numpy()
    indice_numpy = indice_tensor.numpy()
    updates_numpy = updates_tensor.numpy()

    for i in range(1):
        for j in range(5):
            for k in range(indice_tensor[i][1]):
                for l in range(4):
                    var[i][j][indice_tensor[i][0] + k][l] = updates_tensor[i][j][k][l]

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    varOne_numpy.astype(dtype).tofile("./input/input_varRefOne.bin")
    indice_numpy.astype(idx_dtype).tofile("./input/input_indices.bin")
    updates_numpy.astype(dtype).tofile("./input/input_updates.bin")
    varOne_tensor.numpy().tofile("./output/output_golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

