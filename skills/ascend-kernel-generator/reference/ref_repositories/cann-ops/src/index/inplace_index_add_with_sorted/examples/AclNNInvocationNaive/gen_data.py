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
import tensorflow as tf

def gen_golden_data_simple():
    self = np.random.randn(4, 2).astype(tf.bfloat16.as_numpy_dtype)
    index = torch.randint(0, 4, (4,)).to(torch.int64)
    src = np.random.randn(4, 2).astype(tf.bfloat16.as_numpy_dtype)
    dim = 0

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    self.tofile("./input/input_self.bin")
    index.numpy().astype(np.int64).tofile("./input/input_index.bin")
    src.tofile("./input/input_src.bin")

    self_tensor = torch.from_numpy(self.astype(np.float32)).to(torch.bfloat16)
    src_tensor = torch.from_numpy(src.astype(np.float32)).to(torch.bfloat16)
    self_tensor.index_add_(dim, index, src_tensor, alpha=1)
    
    self_tensor.to(torch.float32).numpy().astype(tf.bfloat16.as_numpy_dtype).tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

