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


def mse_loss_grad_v2(input_predict, input_label, input_dout, reduction):
    if reduction == 'mean':
        reduce_elts = 1.0
        for i in input_predict.shape:
            reduce_elts *= i
        cof = (reduce_elts**(-1)) * 2.0
    else:
        cof = 2.0
    
    sub_res = input_predict - input_label
    norm_grad = sub_res * cof
    golden = norm_grad * input_dout
    return golden

def gen_data_and_golden():
    print(">> run example <<")
    reduction = 'mean'
    predict = np.random.uniform(-100, 100, [2, 2]).astype('float32')
    label = np.random.uniform(-100, 100, [2, 2]).astype('float32')
    dout = np.random.uniform(-100, 100, [2, 2]).astype('float32')
    out = mse_loss_grad_v2(predict, label, dout, reduction)
    print("golden out = ", out)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    predict.tofile("./input/predict.bin")
    label.tofile("./input/label.bin")
    dout.tofile("./input/dout.bin")
    out.tofile("./output/out.bin")

if __name__ == "__main__":
    gen_data_and_golden()
