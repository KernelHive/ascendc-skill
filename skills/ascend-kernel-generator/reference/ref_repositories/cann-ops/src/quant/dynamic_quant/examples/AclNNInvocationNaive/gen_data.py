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
    arr = np.array([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=np.float32)
    arr_abs = np.abs(arr)
    row_max = np.amax(arr_abs, axis=2)
    row_scale = row_max / 127.0
    arr = arr / row_scale[..., None]
    golden = np.round(arr).astype(np.int8)
    os.system("mkdir -p output")
    golden.tofile("./output/golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
