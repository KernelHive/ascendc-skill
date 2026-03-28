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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gen_golden_data(case_id):
    if case_id == "case1":
        x_shape = (4980, 4, 2048)
        eps = 1e-5
    elif case_id == "case2":
        x_shape = (512, 4, 20480)
        eps = 1e-7
    else:
        print("[ERROR] The case_id error, please input [case1 / case2]")
        exit(1)
    dtype = torch.float32
    device = "cpu"
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    y = torch.randn(*x_shape, dtype=dtype, device=device)
    gamma = nn.Parameter(torch.randn(x_shape[2], dtype=dtype, device=device))
    beta = nn.Parameter(torch.randn(x_shape[2], dtype=dtype, device=device))
    normalized_shape = (x_shape[-1],)
    add = torch.add(x, y)
    res = F.layer_norm(add, normalized_shape, gamma, beta, eps)
    input_x = x.detach().numpy().astype(np.float32)
    input_y = y.detach().numpy().astype(np.float32)
    input_gamma = gamma.detach().numpy().astype(np.float32)
    input_beta = beta.detach().numpy().astype(np.float32)
    golden = res.detach().numpy().astype(np.float32)
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    input_gamma.tofile("./input/input_gamma.bin")
    input_beta.tofile("./input/input_beta.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data(sys.argv[1])
