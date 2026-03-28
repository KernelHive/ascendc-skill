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
    num_weights = 3
    weight = np.random.randn(9).astype(np.float32).reshape(3, 3)
    indices = np.random.randint(0, num_weights, size=6).astype(np.int64)
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    per_sample_weights = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)

    max_norm = None
    norm_type = 2

    scale_grad_by_freq = False
    mode = 'sum'
    sparse = False
    include_last_offset = False
    padding_idx = 1

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    weight.tofile("./input/input_0.bin")
    indices.tofile("./input/input_1.bin")
    offsets.numpy().astype(np.int64).tofile("./input/input_2.bin")
    per_sample_weights.numpy().astype(np.float32).tofile("./input/input_3.bin")

    weight = torch.from_numpy(weight)
    indices = torch.from_numpy(indices)
    output = torch.nn.functional.embedding_bag(
        indices,
        weight,
        offsets = offsets,
        max_norm = max_norm,
        norm_type = norm_type,
        scale_grad_by_freq = scale_grad_by_freq,
        mode = mode,
        sparse = sparse,
        per_sample_weights = per_sample_weights,
        include_last_offset = include_last_offset,
        padding_idx = padding_idx,
    )

    output.numpy().astype(np.float32).tofile("./output/golden_0.bin")
    print(output)

if __name__ == "__main__":
    gen_golden_data_simple()
