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
    input_shape = [10, 10]
    self = np.random.uniform(-100, 100, input_shape).astype(dtype)
    self_tensor = torch.from_numpy(self)
    out_tensor = torch.histc(self_tensor, bins=10)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    self.astype(dtype).tofile("./input/input_self.bin")
    out_tensor.numpy().tofile("./output/output_golden_values.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

