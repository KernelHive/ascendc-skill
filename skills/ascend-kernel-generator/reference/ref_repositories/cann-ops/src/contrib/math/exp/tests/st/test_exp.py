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


def exp_test(x, base, scale, shift): 
    x = tf.convert_to_tensor(x)
    y = x * scale + shift
    if base != -1.0: 
        if base < 0: 
            return None
        y *= math.log(base)
    return y.numpy()


def calc_expect_func(x, base=2.0, scale=1.0, shift=2.0): 
    """
    calc_expect_func
    """
    res = exp_test(x["x"], base, scale, shift)
    return [res]
