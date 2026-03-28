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

def golden_func(x, gamma, beta):
    remaining_dims = x.shape[2:]
    hw = 1
    for size in remaining_dims:
        hw*=size
    res = torch.ops.aten.native_group_norm(input=torch.from_numpy(x.astype(np.float64)),
                                           weight=torch.from_numpy(gamma.astype(np.float64)),
                                           bias=torch.from_numpy(beta.astype(np.float64)), 
                                           N=N, C=C, HxW=hw, group=G, eps=eps)
    out = res[0]
    meanOut = res[1]
    rstdOut = res[2]

    sigmoid_out = 1 / (1 + torch.exp(-swish_scale * out))
    out = out * sigmoid_out
    return out, meanOut, rstdOut

def gen_golden_data_simple():
    x = np.random.uniform(0, 1, x_shape).astype(np_q_type)
    gamma = np.random.uniform(0, 1, gamma_shape).astype(np_q_type) 
    beta = np.random.uniform(0, 1, beta_shape).astype(np_q_type) 
    os.system("mkdir -p input")
    x.tofile("./input/input_x.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")

    golden = golden_func(x, gamma, beta)
    os.system("mkdir -p output")
    golden[0].numpy().astype(np_q_type).tofile("./output/out.bin")
    golden[1].numpy().astype(np_q_type).tofile("./output/meanOut.bin")
    golden[2].numpy().astype(np_q_type).tofile("./output/rstdOut.bin")


if __name__ == "__main__":
    eps = 0.00001
    swish_scale = 1
    x_shape = [100,32]
    gamma_shape = [32]
    beta_shape = [32]
    N = x_shape[0]
    C = x_shape[1]
    G = 8
    np_q_type = np.float16
    gen_golden_data_simple()