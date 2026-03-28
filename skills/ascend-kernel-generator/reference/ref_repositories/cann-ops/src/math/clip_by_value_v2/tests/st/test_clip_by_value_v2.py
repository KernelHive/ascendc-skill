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


def clip_by_value_v2_test(x, clip_value_min, clip_value_max):
    x_tensor = tf.convert_to_tensor(x)
    clip_value_min_tensor = tf.convert_to_tensor(clip_value_min)
    clip_value_max_tensor = tf.convert_to_tensor(clip_value_max)
    res = tf.clip_by_value(x_tensor, clip_value_min_tensor, clip_value_max_tensor)
    return res.numpy()


def calc_expect_func(x, clip_value_min, clip_value_max, y):
    """
    calc_expect_func
    """
    res = clip_by_value_v2_test(x['value'], clip_value_min['value'], clip_value_max['value'])
    return [res]
