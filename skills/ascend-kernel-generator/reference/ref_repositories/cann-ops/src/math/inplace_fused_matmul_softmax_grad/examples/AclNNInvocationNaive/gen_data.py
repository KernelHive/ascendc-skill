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


def gen_golden_data_simple():
    torch.manual_seed(11)
    softmax_output = torch.randn(12, 20000)
    grad_output = torch.randn(12, 128)
    values = torch.randn(20000, 128)
    grad_softmax = torch.matmul(grad_output, torch.transpose(values, -2, -1))
    grad_x_golden = (grad_softmax - (softmax_output * grad_softmax).sum(-1, keepdim=True)) * softmax_output

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    softmax_output.numpy().tofile("./input/1_softmax_output.bin")
    grad_output.numpy().tofile("./input/2_grad_output.bin")
    values.numpy().tofile("./input/3_values.bin")
    grad_x_golden.numpy().tofile("./output/4_grad_x_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
