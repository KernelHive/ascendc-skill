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
    input_data = torch.randn(4, 2).to(torch.float32)
    index = torch.randint(0, 4, (2,)).to(torch.int32)
    dim = 0
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_data.numpy().astype(np.float32).tofile("./input/input_self.bin")
    index.numpy().tofile("./input/input_index.bin")

    result = torch.index_select(input_data, dim, index)
    
    result.numpy().tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

