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
import torch.nn.functional as F
import numpy as np

def gen_golden_data_simple():
    self = torch.randn(2, 32).to(torch.float32)
    
    dim = -1

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    self.numpy().astype(np.float32).tofile("./input/input_x.bin")
    x = torch.chunk(self, 2, dim=dim)
    x0 = x[0].type(torch.float32)
    x1 = x[1].type(torch.float32)
    result = F.silu(x0) * x1
    
    result.numpy().tofile("./output/output_golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

