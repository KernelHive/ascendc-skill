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
    self_shape = [3, 4]
    index_shape = [2, 3]
    src_shape = [2, 3]

    self_tensor = torch.randn(self_shape).to(torch.float32)
    index_tensor = torch.randint(0, 3, index_shape).to(torch.int64)
    src_tensor = torch.randn(src_shape).to(torch.float32)

    self_numpy = self_tensor.numpy()
    index_numpy = index_tensor.numpy()
    src_numpy = src_tensor.numpy()

    self_tensor.scatter_(0, index_tensor, src_tensor)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    self_numpy.astype(dtype).tofile("./input/input_self.bin")
    index_numpy.astype(idx_dtype).tofile("./input/input_index.bin")
    src_numpy.astype(dtype).tofile("./input/input_src.bin")
    self_tensor.numpy().tofile("./output/output_golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

