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
import torch
import os
def circular_pad_backward(x, paddings, y):
    out = torch.nn.functional.pad(y, paddings, "circular")
    loss = (x * out).sum()
    loss.backward()
    return y.grad

def gen_golden_data_simple():
    dtype = torch.float
    gradoutputshape = (1, 1, 4, 4)
    gradoutputvalues = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    gradintputshape = (1, 1, 2, 2)
    gradintputvalues = [0, 0, 0, 0]
    padding = (1, 1, 1, 1)
    gradoutput = torch.tensor(gradoutputvalues, dtype=torch.float).reshape(gradoutputshape)
    gradoutput.requires_grad = True
    gradinput = torch.tensor(gradintputvalues, dtype=torch.float).reshape(gradintputshape)
    gradinput.requires_grad = True

#调用函数
    out = circular_pad_backward(gradoutput, padding, gradinput)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    gradoutput.requires_grad = False
    gradinput.requires_grad = False
    gradoutput.numpy().tofile("./input/input.bin")
    out.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

