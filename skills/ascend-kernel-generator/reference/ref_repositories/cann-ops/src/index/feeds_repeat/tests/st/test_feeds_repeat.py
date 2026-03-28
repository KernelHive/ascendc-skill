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
import numpy as np


def feeds_repeat_test(feeds, feeds_repeat_times, output_feeds_size):
    y = np.repeat(feeds, feeds_repeat_times, axis = 0)
    total = sum(feeds_repeat_times)
    pad_diff = output_feeds_size - total
    pad_shape = [(0, 0) for _ in range(len(feeds.shape))]
    pad_shape[0] = (0, pad_diff)
    y = np.pad(y, pad_shape, 'constant', constant_values = (0))
    return y


def calc_expect_func(feeds, feeds_repeat_times, y, output_feeds_size):
    res = feeds_repeat_test(feeds["value"], feeds_repeat_times["value"], output_feeds_size)
    return [res]