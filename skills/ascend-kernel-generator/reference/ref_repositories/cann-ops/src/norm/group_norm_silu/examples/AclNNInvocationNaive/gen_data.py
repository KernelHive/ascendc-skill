#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================

import sys
import os
import math
import numpy as np
import torch
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(5)

d_type_dict = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": bf16,
    'int64': np.int64
}


def do_group_norm_silu(x, gamma, beta, group, eps, dtype):
    if dtype == "float16" or dtype == "bfloat16":
        x = x.astype(np.float32)
        gamma = gamma.astype(np.float32)
        beta = beta.astype(np.float32)

    N = x.shape[0]
    C = x.shape[1]
    remaining_dims = x.shape[2:]
    HW = 1
    for size in remaining_dims:
        HW *= size
    
    x = torch.from_numpy(x)
    gamma = torch.from_numpy(gamma)
    beta = torch.from_numpy(beta)
    output = torch.ops.aten.native_group_norm(x, gamma, beta, N, C, HW, group, eps)
    out = output[0]
    sigmoid_out = 1 / (1 + torch.exp(-out))
    out = out * sigmoid_out
    out = out.numpy()

    if dtype == "float16" or dtype == "bfloat16":
        out = out.astype(d_type_dict.get(dtype))
    return out


def gen_input_data(shape, dtype, input_range):
    dtype = d_type_dict.get(dtype)
    x = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    gamma = np.random.uniform(input_range[0], input_range[1], shape[1]).astype(dtype)
    beta = np.random.uniform(input_range[0], input_range[1], shape[1]).astype(dtype)

    workspace = np.array([1])
    return x, gamma, beta, workspace


def gen_golden_data_simple(shape, dtype, input_range, group, eps):

    x, gamma, beta, workspace = gen_input_data(shape, dtype, input_range)
    golden = do_group_norm_silu(x, gamma, beta, group, eps, dtype)
    x.tofile(f"./input/x.bin")
    gamma.tofile(f"./input/gamma.bin")
    beta.tofile(f"./input/beta.bin")
    workspace.tofile(f"./input/workspace.bin")
    golden.tofile(f"./output/golden.bin")


if __name__ == "__main__":
    # 清理bin文件
    dtype = sys.argv[1]
    shape, input_range = [4, 2, 8, 8], [0.1, 1]
    group, eps = 2, 1e-6
    gen_golden_data_simple(shape, dtype, input_range, group, eps)
