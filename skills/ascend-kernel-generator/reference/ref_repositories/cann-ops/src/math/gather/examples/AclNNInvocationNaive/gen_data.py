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
import numpy as np
import tensorflow as tf

def gen_golden_data_simple():
    x = np.random.uniform(-5, 5, [96, 43, 1023]).astype(np.int32)
    indices = np.random.uniform(1, 43, [96, 43]).astype(np.int32)
    validate_indices = True
    batch_dim = 1
    is_preprocessed = False
    negative_index_support = False
    golden = tf.gather(x, indices, batch_dims=batch_dim).numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.tofile("./input/input_x1.bin")
    indices.tofile("./input/input_indices.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

