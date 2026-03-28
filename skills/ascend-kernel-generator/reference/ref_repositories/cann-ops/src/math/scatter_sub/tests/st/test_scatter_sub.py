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


def scatter_sub_test(var, indices, updates, use_locking=False):
    ref = tf.Variable(var)
    indices_tensor = tf.constant(indices)
    updates_tensor = tf.constant(updates)
    scatter_sub_result = tf.raw_ops.ScatterSub(
        ref=ref, 
        indices=indices_tensor, 
        updates=updates_tensor, 
        use_locking=use_locking)
    return scatter_sub_result.numpy()


def calc_expect_func(var, indices, updates, use_locking=False):
    res = scatter_sub_test(var["value"], indices["value"], updates["value"], use_locking)
    return [res]