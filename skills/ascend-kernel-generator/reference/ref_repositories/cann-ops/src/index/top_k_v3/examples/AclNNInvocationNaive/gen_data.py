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
    input_shape = [2, 16]
    output_shape = [2, 2]

    self = np.random.uniform(-1, 1, input_shape).astype(dtype)
    indices = np.random.uniform(-1, 1, output_shape).astype(idx_dtype)
    values = np.random.uniform(-1, 1, output_shape).astype(dtype)
    largest_val = True
    sorted_val = True
    k_val = 2
    dim_val = 1
    self_tensor = torch.from_numpy(self)
    indices_tensor = torch.from_numpy(indices)
    values_tensor = torch.from_numpy(values)
    torch.topk(self_tensor, k=k_val, dim=dim_val, largest=largest_val, sorted=sorted_val, out=(values_tensor, indices_tensor))

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    self.astype(dtype).tofile("./input/input_self.bin")
    values_tensor.numpy().tofile("./output/output_golden_values.bin")
    indices_tensor.numpy().tofile("./output/output_golden_indices.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

