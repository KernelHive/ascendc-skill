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
import logging
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

# 配置logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

tf.disable_v2_behavior()


def gen_golden_data_simple():
    input_var = np.random.uniform(-10, 10, [3, 4, 24, 24]).astype(np.int8)
    input_indices = np.random.uniform(0, 3, [3]).astype(np.int32)
    input_updates = np.random.uniform(-10, 10, [3, 4, 24, 24]).astype(np.int8)
    use_locking = False

        # 打印输入值
    logging.info("Input var:")
    logging.info(input_var[:5])
    logging.info("Input indices:")
    logging.info(input_indices)
    logging.info("Input updates:")
    logging.info(input_updates[:5])

    ref = tf.Variable(input_var)
    indices = tf.constant(input_indices)
    updates = tf.constant(input_updates)
    scatter_sub_op = tf.raw_ops.ScatterSub(ref=ref, indices=indices, updates=updates)
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        golden = sess.run(scatter_sub_op)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_var.tofile("./input/input_var.bin")
    input_indices.tofile("./input/input_indices.bin")
    input_updates.tofile("./input/input_updates.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

