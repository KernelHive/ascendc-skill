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
    input_predict = np.random.uniform(-1, 1, [100, 100]).astype(np.float32)
    input_label = np.random.uniform(-1, 1, [100, 100]).astype(np.float32)
    input_dout = np.random.uniform(-1, 1, [100, 100]).astype(np.float32)
    
    reduction = "mean"
    if 'mean' == reduction:
        reduce_elts = 1.0
        for i in input_predict.shape:
            reduce_elts *= i
        cof = (reduce_elts**(-1)) * 2.0
    else:
        cof = 2.0

    sub_res = input_predict - input_label
    norm_grad = sub_res * cof
    golden = norm_grad * input_dout

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_predict.tofile("./input/input_predict.bin")
    input_label.tofile("./input/input_label.bin")
    input_dout.tofile("./input/input_dout.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
