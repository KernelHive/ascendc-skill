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
    input_x = np.random.uniform(1, 10, [1, 23]).astype(np.float16)
    clip_value_min = np.random.uniform(1, 3, [1]).astype(np.float16)
    clip_value_max = np.random.uniform(4, 10, [1]).astype(np.float16)
    y = torch.clip(torch.from_numpy(input_x), torch.from_numpy(clip_value_min), torch.from_numpy(clip_value_max))

    golden = y.detach().numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    clip_value_min.tofile("./input/input_clip_value_min.bin")
    clip_value_max.tofile("./input/input_clip_value_max.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


