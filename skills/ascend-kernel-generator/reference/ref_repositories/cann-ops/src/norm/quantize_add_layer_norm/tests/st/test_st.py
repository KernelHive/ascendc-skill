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


def _quantize(x, scale, offset, axis):
    x_shape = x.shape
    scale_shape = scale.shape
    if len(x_shape) != len(scale_shape):
        # 支持per tensor，per channel，per head
        tmp_scale_shape = [1] * len(x_shape)
        tmp_scale_shape[axis] = scale_shape[0]
        scale = np.reshape(scale, tmp_scale_shape)

    scale_rst = x / scale
    if offset is not None:
        offset = np.reshape(offset, scale.shape)
        scale_rst = scale_rst + offset

    round_data = np.round(scale_rst, 0)
    round_data = round_data.clip(-128, 127)
    round_data = round_data.astype("int8", copy=False)
    return round_data


def quantize_add_layer_norm(**kwargs):
    tensor_x1 = kwargs.get('x1', {'value': None})['value']
    tensor_x2 = kwargs.get('x2', {'value': None})['value']
    tensor_gamma = kwargs.get('gamma', {'value': None})['value']
    tensor_beta = kwargs.get('beta', {'value': None})['value']
    tensor_bias = kwargs.get('bias', {'value': None})['value']
    tensor_scales = kwargs.get('scales', {'value': None})['value']

    epsilon = 1e-05

    gammashape = tensor_gamma.shape
    x = tensor_x1 + tensor_x2 + tensor_bias
    mean = np.mean(x, -1, keepdims=True)
    tmp = (x - mean) * (x - mean)
    tensor_var = np.mean(tmp, -1, keepdims=True)
    rstd = 1 / np.sqrt(tensor_var + epsilon)
    y = (x - mean) * rstd * tensor_gamma + tensor_beta

    _axis = len(tensor_x1.shape) - len(gammashape)

    y = _quantize(y, tensor_scales, None, _axis)
    return y, x