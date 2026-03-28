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

def gen_golden_data_simple():
    shape_params = (64, 1, 60)
    dtype = np.float16

    input1 = np.random.uniform(0.0, 1.0, size=shape_params).astype(dtype)
    input2 = np.random.uniform(0.0, 1.0, size=shape_params).astype(dtype)
    sel = np.random.choice([True, False], size=shape_params).astype(bool)

    input1_sel = input1 * sel
    input2_sel = input2 * (~sel)
    reduce_res = input1_sel + input2_sel
    max_res = np.max(reduce_res, axis=-1, keepdims=True)
    sub_res = reduce_res - max_res
    exp_res = np.exp(sub_res)
    sum_res = np.sum(exp_res, axis=-1, keepdims=True)
    result = exp_res / sum_res


    os.system("mkdir -p input")
    os.system("mkdir -p output")
    sel.tofile("./input/input_1.bin")
    input1.astype(dtype).tofile("./input/input_2.bin")
    input2.astype(dtype).tofile("./input/input_3.bin")
    result.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

