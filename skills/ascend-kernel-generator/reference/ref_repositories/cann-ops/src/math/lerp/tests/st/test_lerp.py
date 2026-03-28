#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import tensorflow as tf
import numpy as np


def lerp_test(start, end, weight):
    tensor_start = tf.convert_to_tensor(start)
    tensor_end = tf.convert_to_tensor(end)
    tensor_weight = tf.convert_to_tensor(weight)
    res = tensor_start + tensor_weight * (tensor_end - tensor_start)
    return res.numpy()


def calc_expect_func(start, end, weight, y):
    res = lerp_test(start["value"], end["value"], weight["value"])
    return [res]