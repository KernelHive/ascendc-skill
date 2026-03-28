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


def triu_test(x, diagonal):
    tensor = tf.convert_to_tensor(x)
    triu_result = tf.linalg.band_part(tensor, diagonal, -1)
    return triu_result.numpy()


def calc_expect_func(x, y, diagonal=0):
    res = triu_test(x["value"], diagonal)
    return [res]