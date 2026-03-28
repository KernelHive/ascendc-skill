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
    grad = np.random.randn(6).astype(np.float32).reshape(2, 3)
    num_weights = 4
    indices = np.random.randint(0, num_weights, size=2).astype(np.int64)
    padding_idx = 0
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    grad.tofile("./input/input_x.bin")
    indices.tofile("./input/input_y.bin")

    grad = torch.from_numpy(grad)
    indices = torch.from_numpy(indices)
    golden = torch.ops.aten.embedding_dense_backward(
    grad_output=grad,
    indices=indices,
    num_weights=num_weights,
    padding_idx=padding_idx,
    scale_grad_by_freq=False
)

    golden.numpy().astype(np.float32).tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

