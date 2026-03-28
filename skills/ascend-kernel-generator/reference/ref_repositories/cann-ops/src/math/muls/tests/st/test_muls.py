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


def muls_test(x, value):
    x_tensor = tf.convert_to_tensor(x)
    dtype = x_tensor.dtype
    # 新增对复数类型的处理 <button class="citation-flag" data-index="3"><button class="citation-flag" data-index="5">
    if dtype in [tf.complex64, tf.complex128]:
        float_num = tf.constant(value, dtype=tf.float32)
        float_as_complex = tf.complex(float_num, tf.constant(0.0, dtype=tf.float32))  # 输出：2+0j [[2]][[7]]
        res = tf.multiply(x_tensor, float_as_complex)  # 结果：(3*2)+(4*2)j = 6+8j [[2]]
    elif dtype in [tf.int16, tf.int32, tf.int64]:
        value = tf.cast(value, dtype)
        res = x_tensor * value

    else:
        # 处理其他数值类型（如float32/64）
        value = tf.cast(value, dtype)
        res = x_tensor * value
    
    return res.numpy()


def calc_expect_func(x, y, value):
    res = muls_test(x["value"], value)
    return [res]
