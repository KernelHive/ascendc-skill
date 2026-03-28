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


def mse_loss_v2(input_predict, input_target, reduction):
    ori_type = input_predict.dtype
    input_predict = torch.from_numpy(input_predict)
    input_target = torch.from_numpy(input_target)
    output = torch.nn.functional.mse_loss(input_predict, input_target, reduction=reduction).numpy().astype(ori_type)
    if reduction == 'none':
        return output
    golden = np.zeros_like(input_predict.shape)
    golden.ravel()[0] = output
    return golden[0].astype('float32')

def gen_data_and_golden():
    reduction = 'mean'
    predict = np.random.uniform(-100, 100, [2, 2]).astype('float32')
    target = np.random.uniform(-100, 100, [2, 2]).astype('float32')
    print("golden predict = ", predict)
    print("golden target = ", target)
    out = mse_loss_v2(predict, target, reduction)
    print("golden out = ", out)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    predict.tofile("./input/predict.bin")
    target.tofile("./input/target.bin")
    out.tofile("./output/out.bin")

if __name__ == "__main__":
    gen_data_and_golden()
