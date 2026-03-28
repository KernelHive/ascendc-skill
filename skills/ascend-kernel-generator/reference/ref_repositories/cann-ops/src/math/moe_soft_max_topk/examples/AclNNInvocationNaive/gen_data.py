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
import sys
import numpy as np
import torch


def gen_golden_data(case_id):
    if case_id == "case1":
        x_shape = (1024, 16)
        k = 4
    elif case_id == "case2":
        x_shape = (2048, 32)
        k = 6
    dtype = torch.float32
    device = "cpu"
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    y, indices = torch.topk(torch.softmax(x, dim=1), k)

    input_x = x.detach().numpy().astype(np.float32)
    output_y = y.detach().numpy().astype(np.float32)
    output_indices = indices.detach().numpy().astype(np.int32)
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    output_y.tofile("./output/golden_y.bin")
    output_indices.tofile("./output/golden_indices.bin")

if __name__ == "__main__":
    gen_golden_data(sys.argv[1])
