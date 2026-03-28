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
from functools import reduce


def gen_golden_data_simple():
    dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
         19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype=np.float32).reshape(2, 3, 4)
    mean = np.array([6.5, 18.5], dtype=np.float32).reshape(2, 1)
    rstd = np.array([0.2896827, 0.2896827], dtype=np.float32).reshape(2, 1)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
         19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype=np.float32).reshape(2, 3, 4)
    gamma = np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape(3)
    N = dy.shape[0]
    C = dy.shape[1]
    if len(dy.shape) > 2:
        HxW = reduce(lambda x, y: x * y, dy.shape[2:])
    else:
        HxW = 1
    G = mean.shape[-1]
    num_groups = 1
    dx_is_require = True
    dgamma_is_require = True
    dbeta_is_require = True
    dy_tensor = torch.from_numpy(dy)
    mean_tensor = torch.from_numpy(mean)
    rstd_tensor = torch.from_numpy(rstd)
    x_tensor = torch.from_numpy(x)
    gamma_tensor = torch.from_numpy(gamma)
    grad_x, grad_gamma, grad_beta = torch.ops.aten.native_group_norm_backward(dy_tensor, x_tensor, mean_tensor,
                                                                              rstd_tensor, gamma_tensor, N, C, HxW,
                                                                              num_groups,
                                                                              [dx_is_require, dgamma_is_require,
                                                                               dbeta_is_require])
    res_shape = [dy.shape, gamma.shape, gamma.shape]
    res_list = [grad_x, grad_gamma, grad_beta]
    for i, value in enumerate([dx_is_require, dgamma_is_require, dbeta_is_require]):
        if value == True:
            res_list[i] = res_list[i].numpy()
        else:
            res_list[i] = None

    res_list[0].tofile("./output/golden1.bin")
    res_list[1].tofile("./output/golden2.bin")
    res_list[2].tofile("./output/golden3.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
