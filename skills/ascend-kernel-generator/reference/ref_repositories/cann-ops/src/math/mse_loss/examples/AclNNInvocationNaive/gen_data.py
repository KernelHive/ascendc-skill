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
import torch.nn as nn

def gen_golden_data_simple():
    input_predict = np.random.uniform(-1, 1, [100, 100]).astype(np.float32)
    input_label = np.random.uniform(-1, 1, [100, 100]).astype(np.float32)
    # 初始化损失函数
    mse_loss = nn.MSELoss(reduction='mean')

    # 假设我们有一些预测值和目标值
    predictions = torch.tensor(input_predict)
    targets = torch.tensor(input_label)

    # 计算损失
    loss = mse_loss(predictions, targets)

    golden = loss.numpy()

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_predict.tofile("./input/input_predict.bin")
    input_label.tofile("./input/input_label.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
