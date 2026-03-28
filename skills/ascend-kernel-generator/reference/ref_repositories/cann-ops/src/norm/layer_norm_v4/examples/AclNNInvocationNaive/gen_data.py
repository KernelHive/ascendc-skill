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

import numpy as np


def layer_norm_compute(input_x, input_gamma, input_beta, out_dtypes,
                       epsilon):
    import torch
    dtype = input_x.dtype
    gamma_dtype = input_gamma.dtype
    beta_dtype = input_beta.dtype
    if "float16" in str(dtype):
        input_x = input_x.astype("float32")
        input_gamma = input_gamma.astype("float32")
        input_beta = input_beta.astype("float32")
    if dtype == "float32" and (gamma_dtype == "float64" or beta_dtype == "float64"):
        input_x = input_x.astype("float64")
        input_gamma = input_gamma.astype("float64")
        input_beta = input_beta.astype("float64")
    input_tensor = torch.from_numpy(input_x)
    gamma_tensor = torch.from_numpy(input_gamma)
    beta_tensor = torch.from_numpy(input_beta)
    normalized_shape = input_gamma.shape
    res, mean, variance = torch.ops.aten.native_layer_norm(
        input_tensor, normalized_shape, gamma_tensor, beta_tensor, eps=epsilon)
    res = res.numpy()
    variance = variance.numpy()
    mean = mean.numpy()
    out_dtype = "float32"
    return res.astype(out_dtype[0], copy=False), mean, variance


def gen_golden_data_simple():
    input_x = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2 , 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2, 2, 2], dtype=np.float32).reshape(1, 2, 32)
    input_gamma = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape(32)
    input_beta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0,
                         0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0], dtype=np.float32).reshape(32)
    epsilon = 1e-5

    out_dtypes = "float32"
    res, mean, variance = layer_norm_compute(input_x, input_gamma, input_beta, out_dtypes, epsilon=epsilon)

    res.tofile("./output/golden1.bin")
    mean.tofile("./output/golden2.bin")
    variance.tofile("./output/golden3.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

