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
from torch.nn import functional as F


def grid_sampler_2d_golden():

    x_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]).reshape(1, 1, 5, 8).type(torch.float32)
    grid_tensor = torch.tensor([-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 
                               1, 0, -1, 1, 0, 1, 1, 1]).reshape(1, 3, 3, 2).type(torch.float32)
    interpolation_mode = "bilinear"
    padding_mode = "zeros"
    align_corners = False
    x_tensor.requires_grad = True
    grid_tensor.requires_grad = True

    y_tensor = F.grid_sample(x_tensor, grid_tensor, interpolation_mode, padding_mode, align_corners)
    loss = torch.sum(y_tensor)
    loss.backward()

    res = x_tensor.grad.numpy()
    res.tofile("./output/golden1.bin")
    res = grid_tensor.grad.numpy()
    res.tofile("./output/golden2.bin")

if __name__ == "__main__":
    grid_sampler_2d_golden()

