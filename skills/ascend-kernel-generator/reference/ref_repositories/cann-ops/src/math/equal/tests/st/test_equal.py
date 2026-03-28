# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================
import tensorflow as tf
import numpy as np


def equal_test(x, y):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    res = tf.equal(x, y)
    return res.numpy().astype(np.bool_)


def calc_expect_func(x1, x2, y):
    """
    calc_expect_func
    """
    res = equal_test(x1["value"], x2["value"])
    return [res]
