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
import numpy as np
import torch


def gen_golden_data_simple():
    num_rows = 18
    num_columns = 10
    batch_shape = [3, 4]
    if num_columns != 0:
        res_shape = [*batch_shape, num_rows, num_columns]
        res = torch.eye(n=num_rows, m=num_columns)
    else:
        res_shape = [*batch_shape, num_rows, num_rows]
        res = torch.eye(n=num_rows)
    dtype = 0
    
    
    res = torch.broadcast_to(res, res_shape)
    if dtype == 0:
        golden = res.numpy().astype(np.float32)
        input_x = np.zeros(res_shape).astype(np.float32)
    elif dtype == 1:
        golden = res.numpy().astype(np.float16)
        input_x = np.zeros(res_shape).astype(np.float16)
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
