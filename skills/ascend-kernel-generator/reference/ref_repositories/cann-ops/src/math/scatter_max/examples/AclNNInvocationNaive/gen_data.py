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

import os
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

def gen_golden_data_simple():

    input_var = np.random.uniform(-10, 10, [256]).astype(np.float16)
    input_indices = np.random.uniform(0, 200, [5]).astype(np.int32)

    input_updates = np.random.uniform(-10, 10, [5]).astype(np.float16)
    use_locking = False

    ref = tf.Variable(input_var)
    indices = tf.constant(input_indices)
    updates = tf.constant(input_updates)
    scatter_max_op = tf.scatter_max(ref=ref, indices=indices, updates=updates)
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        golden = sess.run(scatter_max_op)


    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_var.tofile("./input/input_var.bin")
    input_indices.tofile("./input/input_indices.bin")
    input_updates.tofile("./input/input_updates.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

