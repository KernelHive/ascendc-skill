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
    x_shape = [1, 1, 4, 4, 4]
    z_shape = [1, 1, 2, 2, 2]
    padding = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int64)
    grad_output_np = np.random.uniform(0, 5, x_shape).astype(np.float32)
    input_tensor_np = np.zeros(z_shape).astype(np.float32)
    grad_output = torch.from_numpy(grad_output_np.astype(np.float32))
    input_tensor = torch.from_numpy(input_tensor_np.astype(np.float32))

    grad_input = torch.ops.aten.replication_pad3d_backward(grad_output, input_tensor, padding)
    golden = grad_input.numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    grad_output_np.astype(np.float32).tofile("./input/input_grad_output.bin")
    input_tensor_np.astype(np.float32).tofile("./input/input_self.bin")

    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()