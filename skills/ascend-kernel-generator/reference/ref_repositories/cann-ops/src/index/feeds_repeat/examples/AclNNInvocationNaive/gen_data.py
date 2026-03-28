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

def gen_golden_data_simple():
    dtype = np.float32
    feeds = np.array([1, 2, 3, 4, 5, 6], dtype = dtype).reshape(2, 3)
    feeds_repeat_times = np.array([100, 200], dtype = np.int32)
    output_feeds_size = 500
    
    y = np.repeat(feeds, feeds_repeat_times, axis = 0)
    total = sum(feeds_repeat_times)
    pad_diff = output_feeds_size - total
    pad_shape = [(0, 0) for _ in range(len(feeds.shape))]
    pad_shape[0] = (0, pad_diff)
    golden = np.pad(y, pad_shape, 'constant', constant_values = (0))

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    feeds.astype(dtype).tofile("./input/input_feeds.bin")
    feeds_repeat_times.astype(np.int32).tofile("./input/input_feeds_repeat_times.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

