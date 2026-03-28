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
import torch


def gen_golden_data_simple():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(4, 2)
    scale = np.array([1, 2, 1, 2, 1, 2], dtype=np.float32).reshape(3, 2)
    group_index = np.array([1, 2, 4], dtype=np.int32).reshape(3)
    offset = np.array([2], dtype=np.float32).reshape(1)

    attr_dst_type = 2

    dim_h = x.shape[1]
    if attr_dst_type == 29:
        assert dim_h % 2 == 0, "For output y, if datatype is int4, dim of last axis should be even number"
    dim_e = scale.shape[0]
    assert dim_e > 0, "the first dim of scale shape should be greater than 0"

    x_fp32 = x.astype('float32')
    scale_fp32 = scale.astype('float32')
    offset_fp32 = offset.astype('float32') if offset is not None else None
    y_fp32 = np.empty(shape=(0, dim_h), dtype='float32')

    for row_scale in range(dim_e):
        x_start_row = 0 if row_scale == 0 else group_index[row_scale - 1]
        x_end_row = group_index[row_scale]
        if x_start_row < x_end_row:
            y_rows = x_fp32[x_start_row: x_end_row] * scale_fp32[row_scale]
            if offset is not None:
                y_rows = y_rows + offset_fp32
            y_fp32 = np.concatenate([y_fp32, y_rows], axis=0)
    y_round = np.round(y_fp32, 0)

    if attr_dst_type == 2:
        res = np.clip(y_round, -128, 127).astype("int8")
    elif attr_dst_type == 29:
        res = np.clip(y_round, -8, 7).astype("int4")
    else:
        raise Exception("attr dst_type only support 2(int8) or 29(int4)")

    res.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
