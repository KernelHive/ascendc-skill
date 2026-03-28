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
import copy
import numpy as np


def gen_golden_data_simple():
    data_format = "ND"

    data_x = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(1, 2, 4)
    data_weight = np.array([1, 1], dtype=np.float32).reshape(2)
    data_bias = np.array([0, 0], dtype=np.float32).reshape(2)
    # running_mean and running_var are updated in place while BN is running.
    tmp_running_mean = np.array([0, 0], dtype=np.float32).reshape(2)
    tmp_running_var = np.array([1, 1], dtype=np.float32).reshape(2)

    x_dtype = data_x.dtype

    # TTK will run golden first, so we must not change the original running_mean and running_var.
    data_running_mean = copy.deepcopy(tmp_running_mean)
    data_running_var = copy.deepcopy(tmp_running_var)

    promote_dtype = "float32" if x_dtype.name in ("bfloat16", "float16") else "float64"
    data_x = data_x.astype(promote_dtype)
    data_weight = data_weight.astype(promote_dtype)
    data_bias = data_bias.astype(promote_dtype)
    data_running_mean = data_running_mean.astype(promote_dtype)
    data_running_var = data_running_var.astype(promote_dtype)

    momentum = 0.1
    eps = 1e-5
    is_training = True
    
    tensor_x = torch.from_numpy(data_x)
    tensor_weight = torch.from_numpy(data_weight)
    tensor_bias = torch.from_numpy(data_bias)
    tensor_running_mean = torch.from_numpy(data_running_mean)
    tensor_running_var = torch.from_numpy(data_running_var)

    if data_format == "NHWC":
        tensor_x = tensor_x.permute(0, 3, 1, 2)
    if data_format == "NDHWC":
        tensor_x = tensor_x.permute(0, 4, 1, 2, 3)

    res = torch.ops.aten.native_batch_norm(input=tensor_x, weight=tensor_weight, bias=tensor_bias,
                                           running_mean=tensor_running_mean, running_var=tensor_running_var,
                                           training=is_training, momentum=momentum, eps=eps)

    output = res[0]
    if data_format == "NHWC":
        output = output.permute(0, 2, 3, 1)
    if data_format == "NDHWC":
        output = output.permute(0, 2, 3, 4, 1)

    running_mean = tensor_running_mean.numpy()
    running_var = tensor_running_var.numpy()
    save_mean = res[1].numpy()
    save_rstd = res[2].numpy()
    output.numpy().astype(x_dtype, copy=False).tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

