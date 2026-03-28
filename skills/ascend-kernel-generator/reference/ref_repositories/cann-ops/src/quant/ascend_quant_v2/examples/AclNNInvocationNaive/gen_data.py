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
    scale = np.array([1, 2], dtype=np.float32).reshape(2)
    offset = np.array([1, 2], dtype=np.float32).reshape(2)
    sqrt_mode = False
    round_mode = "round"

    scale_rst = x * (scale ** 2) if sqrt_mode else x * scale
    if offset is not None:
        scale_rst = scale_rst + offset

    if round_mode == "round":
        round_data = np.round(scale_rst, 8)
    elif round_mode == "floor":
        round_data = np.floor(scale_rst)
    elif round_mode == "ceil":
        round_data = np.ceil(scale_rst)
    else:
        round_data = np.trunc(scale_rst)

    round_data = np.clip(round_data, -128, 127)
    round_data = round_data.astype("int8", copy=False)

    round_data.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
