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
    input_shape = [1,1,2,2,2]

    output_size = [1,1,1]
    
    input_self = np.random.uniform(1, 10, input_shape).astype(dtype)
    # 将 NumPy 数组转换为 PyTorch 张量
    input_self_tensor = torch.tensor(input_self, dtype=torch.float32)

    m = torch.nn.AdaptiveMaxPool3d(output_size, True)
    output, indices = m(input_self_tensor)

    gradInput = torch.ops.aten.adaptive_max_pool3d_backward(output, input_self_tensor ,indices)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_self.tofile("./input/self.bin")
    output.numpy().tofile("./input/out.bin")
    indices.to(torch.int32).numpy().tofile("./input/indices.bin")
    gradInput.numpy().tofile("./output/golden_gradInput.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

