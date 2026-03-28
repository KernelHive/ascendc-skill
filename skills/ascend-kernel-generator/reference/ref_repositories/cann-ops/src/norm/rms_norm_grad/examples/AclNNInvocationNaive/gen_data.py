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


def gen_golden_data_simple():
    y_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             24, 25, 26, 27, 28, 29, 30, 31, 32]).reshape(2, 1, 16)
    x_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             24, 25, 26, 27, 28, 29, 30, 31, 32]).reshape(2, 1, 16)
    rstd_tensor = torch.tensor([1, 2]).reshape(2, 1, 1)
    gamma_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).reshape(16)
    s_dim = y_tensor.shape[-1]

    x_cpu = x_tensor.type(torch.float32)
    rstd_cpu = rstd_tensor.type(torch.float32)
    y_cpu = y_tensor.type(torch.float32)
    gamma_cpu = gamma_tensor.type(torch.float32)

    x_rstd = (x_cpu * rstd_cpu)
    dy_rstd = (y_cpu * x_rstd)
    dgamma_golden = torch.sum(dy_rstd.reshape([-1, s_dim]), dim=0, keepdim=True).to(torch.float32)

    dx_golden = ((gamma_cpu * y_cpu).float() * rstd_cpu - torch.mean(
            (gamma_cpu * y_cpu).float() * rstd_cpu.pow(3) * x_cpu, dim=-1, keepdim=True) * x_cpu).to(torch.float32)

    dgamma_golden.numpy().tofile("./output/golden1.bin")
    dx_golden.numpy().tofile("./output/golden2.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

