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


def cross_test(x1, x2):
    x1 = tf.convert_to_tensor(x1, name='x1')
    x2 = tf.convert_to_tensor(x2, name='x2')

    x11, x12, x13 = tf.split(x1, 3, axis=-1)
    x21, x22, x23 = tf.split(x2, 3, axis=-1)

    res =tf.concat([
        (x12 * x23) - (x13 * x22),
        (x13 * x21) - (x11 * x23),
        (x11 * x22) - (x12 * x21)
    ], axis=-1)
    return res.numpy()


def calc_expect_func(x1, x2, y):
    res = cross_test(x1["value"], x2["value"])
    return [res]