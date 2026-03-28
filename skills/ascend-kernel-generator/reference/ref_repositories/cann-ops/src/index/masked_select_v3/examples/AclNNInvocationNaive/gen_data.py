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
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch.int32)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    result = torch.masked_select(input_tensor, mask)

    input_numpy = input_tensor.numpy()
    mask_numpy = mask.numpy()
    result_numpy = result.numpy()

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_numpy.tofile("./input/input.bin")
    mask_numpy.tofile("./input/input_mask.bin")
    result_numpy.tofile("./output/output_golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

