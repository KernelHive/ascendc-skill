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
import torch
import numpy as np
import tensorflow as tf
def reverse_sequence_golden(x, seq_lengths, y, seq_dim, batch_dim):
    if seq_dim < 0:
        seq_dim += len(x.shape)
    if batch_dim < 0:
        batch_dim += len(x.shape)

    tf.compat.v1.disable_eager_execution()
    y = tf.reverse_sequence(x, seq_lengths, seq_axis=seq_dim, batch_axis=batch_dim)
    with tf.compat.v1.Session() as sess:
        golden_y = sess.run(y)
    return [golden_y]

def gen_golden_data_simple():
    x = np.random.rand(4, 8, 8).astype(np.float32)  # [batch, seq_len, feature]
    seq = np.random.randint(1, 8, size=(4,), dtype=np.int32)   # 长度=batch_size
    y = np.zeros((4, 8, 8), dtype=np.float32)

    golden_list = reverse_sequence_golden(x, seq, y, 1, 0)
    golden = np.array(golden_list, dtype=np.float32)  

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(np.float32).tofile("./input/input_x.bin")
    seq.astype(np.int32).tofile("./input/input_seq.bin")
    y.astype(np.float32).tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

