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
import numpy as np


def layer_norm_test(x, y, gamma, beta, epsilon):
    x_tensor = torch.as_tensor(x)
    y_tensor = torch.as_tensor(y)
    gamma_tensor = torch.as_tensor(gamma)
    beta_tensor = torch.as_tensor(beta)
    eps = epsilon
    x_tensor = torch.add(x_tensor, y_tensor)
    res = torch.nn.functional.layer_norm(x_tensor, x_tensor.shape[-1:], gamma_tensor, beta_tensor, eps)
    return res.numpy()


def calc_expect_func(x, y, gamma, beta, epsilon):
    """
    calc_expect_func
    """
    res = layer_norm_test(x['value'], y['value'], gamma['value'], beta['value'], epsilon)
    return [res]
